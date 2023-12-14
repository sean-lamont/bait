from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from loguru import logger

from refactor.dpo.model import RetrievalAugmentedGenerator, DPOTrainModule
from refactor.proof_node import *


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, premises):
        return

    @abstractmethod
    def train(self):
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
            retriever_args={'file_path': path,
                            'theorem_full_name': theorem.full_name,
                            'theorem_pos': position}
        )
        return tactics


# todo better loading for LoRA
# todo add non-generative models
def get_tac_model(config, device):
    if config.model == 'reprover':
        # tac_gen = RetrievalAugmentedGenerator.load(
        #     config.ckpt_path, device=device, freeze=True
        # )
        tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
        if tac_gen.retriever is not None:
            assert config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.indexed_corpus_path)

        tac_gen.freeze()

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

    # todo non-generative models..
    # todo retriever
    
    
    
    # todo adapt holist:

# tactics = io_util.load_tactics_from_file(
#     str(options.path_tactics), str(options.path_tactics_replace))
#
# if options.action_generator_options.asm_meson_no_params_only:
#     logging.warning('Note: Using Meson action generator with no parameters.')
#     action_gen = action_generator.MesonActionGenerator()
#
# else:
#     predictor = get_predictor(options, config)
#     emb_store = None
#
#     if options.HasField('theorem_embeddings'):
#         emb_store = embedding_store.TheoremEmbeddingStore(predictor)
#         emb_store.read_embeddings(str(options.theorem_embeddings))
#         assert emb_store.thm_embeddings.shape[0] == len(theorem_database.theorems)
#
#     action_gen = action_generator.ActionGenerator(
#         theorem_database, tactics, predictor, options.action_generator_options,
#         options.model_architecture, emb_store)
