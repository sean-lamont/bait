from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import ray

from models.end_to_end.tactic_models.tac_models_ import TacModel

warnings.filterwarnings('ignore')

from experiments.end_to_end.common import Context
from models.end_to_end.tactic_models.generator.model import RetrievalAugmentedGenerator


# wrapper to add the retrieval augmented state to the goal node, and to call tac model with Ray
class ReProverWrapper(TacModel):
    def __init__(self, config):
        super().__init__()
        self.distributed = config.distributed

        device = config.device

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RetrievalAugmentedGenerator.load(  # todo replace with RAGLarge or any other model
                config.ckpt_path, device=device, freeze=True
            )
        else:
            tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            self.retriever = True
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)
        else:
            self.retriever = False

        if self.distributed:
            self.tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
        else:
            self.tac_model = ReProverTacGen(tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)

    def get_tactics(self, goal, premises):
        if self.disributed:
            tactics, new_state = ray.get(self.tac_model.get_tactics.remote(goal.goal, premises))
        else:
            tactics, new_state = self.tac_model.get_tactics(goal.goal, premises)

        # save retrieved data to node for retrieval models
        if self.retriever:
            if hasattr(goal, 'data'):
                goal.data['augmented_state'] = new_state
            else:
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
