from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from refactor.dpo.model import RetrievalAugmentedGenerator, DPOTrainModule
from refactor.proof_node import *
import ray


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


def get_tac_model(config, device):
    if config.model == 'reprover':
        # tac_gen = RetrievalAugmentedGenerator.load(
        #     config.ckpt_path, device=device, freeze=True
        # )
        tac_gen = RetrievalAugmentedGenerator(config.config)
        if tac_gen.retriever is not None:
            assert config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.indexed_corpus_path)

        if config.distributed:
            return ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(ReProverTacGen).remote(
                tac_model=tac_gen)
        else:
            return ReProverTacGen(tac_model=tac_gen)

    # todo DPO trained model