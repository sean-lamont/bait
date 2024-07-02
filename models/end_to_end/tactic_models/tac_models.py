from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import ray
import torch
from loguru import logger

from experiments.TacticZero.tactic_zero_data_module import *
from models.TacticZero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.embedding_models.gnn.formula_net.formula_net import FormulaNetEdges
from models.end_to_end.tactic_models.tacticzero.model import TacticZeroTacModel
from models.get_model import get_model

warnings.filterwarnings('ignore')

from data.HOList.utils import io_util
from experiments.end_to_end.common import Context
from models.end_to_end.tactic_models.generator.model import RetrievalAugmentedGenerator
from models.end_to_end.tactic_models.causal_generator.model import RetrievalAugmentedGenerator as RAGLarge
from models.end_to_end.tactic_models.holist_model import holparam_predictor
from models.end_to_end.tactic_models.holist_model import embedding_store
from models.end_to_end.tactic_models.holist_model import action_generator
from experiments.end_to_end.proof_node import *


# todo tidy
class TacModel:
    @abstractmethod
    def get_tactics(self, goals, premises):
        return


class TacWrapper(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goal, premises):
        tactics = ray.get(self.tac_model.get_tactics.remote(goal, premises))

        return tactics


class ReProverWrapper(TacModel):
    def __init__(self, tac_model, retriever=False):
        super().__init__()
        self.tac_model = tac_model
        self.retriever = retriever

    def get_tactics(self, goal, premises):
        tactics, new_state = ray.get(self.tac_model.get_tactics.remote(goal.goal, premises))

        # save retrieved data to node for retrieval models
        if self.retriever:
            goal.data = {'augmented_state': new_state}

        return tactics


# todo make system agnostic
class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, goal, premises):
        path, theorem, position = premises

        tactics, new_state = self.tac_model.generate(
            state=goal,
            num_samples=self.num_sampled_tactics,
            retriever_args=Context(path=path, theorem_full_name=theorem.full_name, theorem_pos=position,
                                   state=goal)
        )

        return tactics, new_state


class HOListTacGen(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goals, premises):
        tactics = self.tac_model.get_tactics([g.goal for g in goals], premises)
        return tactics


class HOL4TacGen(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goals, premises):
        tactics = self.tac_model.get_tactics([g.goal for g in goals], premises)
        return tactics


def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def load_pretrained_encoders(self, encoder_premise, encoder_goal):
    ckpt_dir = self.config.pretrain_ckpt
    ckpt = torch.load(ckpt_dir)['state_dict']
    encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
    encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))


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
            # return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(ReProverTacGen).remote(
            #     tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
            return ReProverWrapper(tac_model, retriever=tac_gen.retriever is not None)

        else:
            return ReProverTacGen(tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)

    if config.model == 'reprover_large':

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RAGLarge.load(
                config.ckpt_path, device=device, freeze=True
            )

        else:
            tac_gen = RAGLarge(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)

        if config.distributed:
            # return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(ReProverTacGen).remote(
            #     tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
            return ReProverWrapper(tac_model, retriever=tac_gen.retriever is not None)

        else:
            return ReProverTacGen(tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)

    elif config.model == 'tacticzero':
        pretrain = config.pretrain

        # default policy models
        context_net = ContextPolicy()
        tac_net = TacPolicy(len(config.tac_config.tactic_pool))
        arg_net = ArgPolicy(len(config.tac_config.tactic_pool), config.model_config.model_attributes.embedding_dim)
        term_net = TermPolicy(len(config.tac_config.tactic_pool), config.model_config.model_attributes.embedding_dim)

        induct_net = FormulaNetEdges(config.model_config.model_attributes.vocab_size,
                                     config.model_config.model_attributes.embedding_dim,
                                     num_iterations=3, global_pool=False,
                                     batch_norm=False)

        encoder_premise = get_model(config.model_config)
        encoder_goal = get_model(config.model_config)

        if pretrain:
            logger.info("Loading pretrained encoder models..")
            load_pretrained_encoders(encoder_premise, encoder_goal)

        tac_model = TacticZeroTacModel(goal_net=context_net,
                                       tac_net=tac_net,
                                       arg_net=arg_net,
                                       term_net=term_net,
                                       induct_net=induct_net,
                                       encoder_premise=encoder_premise,
                                       encoder_goal=encoder_goal,
                                       config=config
                                       )

        if config.distributed:
            tac_gen = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(HOL4TacGen).remote(
                tac_model=tac_model)
            return TacWrapper(tac_gen)
        else:
            return HOL4TacGen(tac_model=tac_model)

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
                predictor = holparam_predictor.HolparamPredictor(config.ckpt_path, config=config)

            elif model_arch == 'PARAMETERS_CONDITIONED_ON_TAC':
                predictor = holparam_predictor.TacDependentPredictor(
                    config.ckpt_path, config=config)
            else:
                raise NotImplementedError

            emb_store = None

            if hasattr(config, 'theorem_embeddings'):
                emb_path = str(config.theorem_embeddings)

                emb_store = embedding_store.TheoremEmbeddingStore(predictor)

                if not os.path.exists(emb_path):
                    logger.info(
                        f'theorem_embeddings file {emb_path} does not exist, computing & saving.',
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
            tac_gen = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(HOListTacGen).remote(
                tac_model=action_gen)
            return TacWrapper(tac_gen)

        else:
            return HOListTacGen(tac_model=action_gen)
