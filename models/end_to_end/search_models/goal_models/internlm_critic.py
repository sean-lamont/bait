# import pytorch_lightning as pl
import torch
from transformers import AutoModel, AutoTokenizer

import lightning.pytorch as pl

torch.set_float32_matmul_precision("medium")


class InternLMCritic:
    def __init__(self, config, device) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(
            config.model_name,
            # device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    def run(self, goals):

        ret = []
        for goal in goals:
            chat = [
                {"role": "user", "content": "Which state is closer to 'no goals'?"},
                {"role": "assistant", "content": goal}
            ]

            ret.append(self.model.get_score(self.tokenizer, chat))

        return ret

