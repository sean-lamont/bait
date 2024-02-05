from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import ray
from loguru import logger

from data.HOList.utils import io_util
from experiments.end_to_end.common import Context
from models.end_to_end.tactic_models.dpo.model import DPOTrainModule
from models.end_to_end.tactic_models.generator.model import RetrievalAugmentedGenerator
from models.end_to_end.tactic_models.holist_model import holparam_predictor
from models.end_to_end.tactic_models.holist_model import embedding_store
from models.end_to_end.tactic_models.holist_model import action_generator
from experiments.end_to_end.proof_node import *


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, premises):
        return


# todo make tac_gen and retriever more system agnostic
class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, goals, premises):
        path, theorem, position = premises

        tactics = self.tac_model.generate(
            state=goals,
            num_samples=self.num_sampled_tactics,
            retriever_args=Context(path=path, theorem_full_name=theorem.full_name, theorem_pos=position, state=goals),
            # {'file_path': path,
            #                 'theorem_full_name': theorem.full_name,
            #                 'theorem_pos': position}
        )
        return tactics


class HOListTacGen(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goals, premises):
        tactics = self.tac_model.get_tactics(goals, premises)
        return tactics


# todo better/more efficient loading for LoRA models.

def get_tac_model(config, device):
    if config.model == 'reprover':

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RetrievalAugmentedGenerator.load(
                config.ckpt_path, device=device, freeze=True
            )

        else:
            tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)

        if config.distributed:
            return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(ReProverTacGen).remote(
                tac_model=tac_gen)
        else:
            return ReProverTacGen(tac_model=tac_gen)

    elif config.model == 'dpo':
        logger.info('Using DPO model..')
        tac_gen = DPOTrainModule.load(
            config.ckpt_path, device=device, freeze=True
        )

        if tac_gen.retriever is not None:
            assert config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.indexed_corpus_path)

        tac_gen.freeze()

        if config.distributed:
            return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(ReProverTacGen).remote(
                tac_model=tac_gen)
        else:
            return ReProverTacGen(tac_model=tac_gen)

    # todo
    elif config.model == 'holist':
        theorem_database = io_util.load_theorem_database_from_file(
            str(config.path_theorem_database))

        tactics = io_util.load_tactics_from_file(
            str(config.path_tactics), str(config.path_tactics_replace))

        if config.action_generator_options.asm_meson_no_params_only:
            logger.warning('Note: Using Meson action generator with no parameters.')
            action_gen = action_generator.MesonActionGenerator()

        else:
            """Returns appropriate predictor based on prover options."""
            model_arch = config.model_architecture

            if model_arch == 'PAIR_DEFAULT':
                predictor = holparam_predictor.HolparamPredictor(str(config.ckpt_path), config=config)

            elif model_arch == 'PARAMETERS_CONDITIONED_ON_TAC':
                predictor = holparam_predictor.TacDependentPredictor(
                    str(config.ckpt_path), config=config)
            else:
                raise NotImplementedError

            emb_store = None

            if hasattr(config, 'theorem_embeddings'):
                emb_path = str(config.theorem_embeddings)

                emb_store = embedding_store.TheoremEmbeddingStore(predictor)

                if not os.path.exists(emb_path):
                    logger.info(
                        'theorem_embeddings file "%s" does not exist, computing & saving.',
                        emb_path)

                    emb_store.compute_embeddings_for_thms_from_db_file(
                        str(config.path_theorem_database))

                    emb_store.save_embeddings(emb_path)
                else:
                    emb_store.read_embeddings(str(config.theorem_embeddings))
                    assert emb_store.thm_embeddings.shape[0] == len(theorem_database.theorems)

            action_gen = action_generator.ActionGenerator(
                theorem_database, tactics, predictor, config.action_generator_options,
                config.model_architecture, emb_store)

        if config.distributed:
            return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(HOListTacGen).remote(
                tac_model=action_gen)
        else:
            return HOListTacGen(tac_model=action_gen)
