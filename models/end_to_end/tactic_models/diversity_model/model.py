"""Lightning module for the tactic generator."""
from typing import List
from typing import Tuple

import torch
import torch.nn.functional as F
from dppy.finite_dpps import FiniteDPP
from loguru import logger
from transformers import T5EncoderModel, AutoTokenizer

torch.set_float32_matmul_precision("medium")


# todo right now, must modify the DPP library code from np.float to float for version compatibility
# aldo modified to remove print statements in compute_K
class DiversityModel(torch.nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.device = device
        self.encoder, self.tokenizer = self.load_encoder(config)
        self.max_seq_len = config.max_seq_len
        self.autoencoder = config.autoencoder if hasattr(config, 'autoencoder') else False
        self.score_network = self.load_score_network(config) if hasattr(config, 'score_network') else None

        self.error_weight = config.error_weight if hasattr(config, 'error_weight') else 1
        self.time_weight = config.time_weight if hasattr(config, 'time_weight') else 1

    def load_score_network(self, config):
        score_network = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.d_model, self.encoder.config.d_model // 2),
            torch.nn.LayerNorm(self.encoder.config.d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder.config.d_model // 2, 2),
        )

        ckpt = torch.load(config.ckpt_dir)
        state_dict = {k[14:]: v for k, v in ckpt.items() if k.startswith('score_network')}

        score_network.load_state_dict(state_dict)

        return score_network

    def load_encoder(self, config):
        if config.ckpt_dir:
            ckpt = torch.load(config.ckpt_dir)
            state_dict = {k[12:]: v for k, v in ckpt.items() if k.startswith('tac_encoder')}

            tac_encoder = T5EncoderModel.from_pretrained(config.model,
                                                         state_dict=state_dict).to(self.device)
        else:
            tac_encoder = T5EncoderModel.from_pretrained(config.model).to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(config.model)

        return tac_encoder, tokenizer

    def get_autoencoder_encoding(self, tactic_ids, tactic_mask):
        hidden_states = self.encoder(tactic_ids, tactic_mask, return_dict=True).last_hidden_state

        # Masked average.
        lens = tactic_mask.sum(dim=1)
        features = (hidden_states * tactic_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def get_tac_encoding(self, goal_ids, goal_mask, tactic_lens):
        # encode all tokens with tactic included
        combined_enc = self.encoder(goal_ids, goal_mask, return_dict=True).last_hidden_state

        # get the tactic embeddings and mean pool them using the provided lengths
        tac_enc = []

        for i in range(combined_enc.shape[0]):
            enc = combined_enc[i, :tactic_lens[i]]
            enc = enc.sum(dim=0) / tactic_lens[i]
            enc = F.normalize(enc, dim=0)
            tac_enc.append(enc)

        tac_enc = torch.stack(tac_enc, dim=0)  # .unsqueeze(1)
        return tac_enc

    # get top p tactics, where p is the cumulative probability of the top k tactics, such that the sum of the probabilities of the top k tactics is greater than or equal to p
    # assume that probs is sorted in ascending order, and that p is between 0 and 1
    def top_p(self, probs, p):
        # take 1 - p since probs is sorted in ascending order
        p = 1 - p
        s = 0
        for i in range(len(probs)):
            s += probs[i]
            if s >= p:
                return len(probs) - i

    def filter_tacs(self, tactics: List[Tuple[str, float]], num_filtered: int, state, theorem, temperature=1.,
                    scale=1., p=0.9):
        with torch.no_grad():
            encs = []

            logprobs = [t[1] / temperature for t in tactics]

            # get softmax over logprobs
            probs = torch.softmax(torch.tensor(logprobs), dim=0) * scale

            # chunking gives slight speedup, but high memory cost
            chunk_size = 1
            for ind in range(0, len(tactics), chunk_size):
                t = [t[0] for t in tactics[ind:ind + chunk_size]]

                goals = [t_ + theorem + '\n\n' + state for t_ in t]

                tokenized_goals = self.tokenizer(
                    goals,
                    padding="longest",
                    max_length=int(self.max_seq_len * 1.5),
                    truncation=True,
                    return_tensors="pt", )

                tokenized_tactics = self.tokenizer(
                    t,
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )

                lens = tokenized_tactics.attention_mask.sum(dim=1)

                if not self.autoencoder:
                    enc = self.get_tac_encoding(tokenized_goals.input_ids.to(self.device),
                                                tokenized_goals.attention_mask.to(self.device), lens.to(self.device))
                else:
                    enc = self.get_autoencoder_encoding(tokenized_tactics.input_ids.to(self.device),
                                                        tokenized_tactics.attention_mask.to(self.device))

                # enc = enc.squeeze(1)

                encs.append(enc)

            vec_matrix = torch.cat(encs, dim=0)

            # augment probs by time/error scores
            if self.score_network:
                scores = self.score_network(vec_matrix)

                error_preds = torch.sigmoid(scores[:, 0])
                time_scores = scores[:, 1]

                time_scores = 1 - torch.softmax(time_scores, dim=0)
                probs = probs + self.error_weight * error_preds + self.time_weight * time_scores

            # dynamic number of tactics to filter, based on the eigenvalues of the similarity matrix
            sim_matrix = vec_matrix @ vec_matrix.T
            sim_matrix = sim_matrix.cpu().numpy()

            DPP = FiniteDPP('likelihood', **{'L': sim_matrix})

            DPP.compute_K(msg=True)
            k_sum = sum(DPP.K_eig_vals)
            k = self.top_p(DPP.K_eig_vals / k_sum, p)

            if k > num_filtered:
                num_filtered = k

            if num_filtered >= len(tactics):
                return [[i for i in range(len(tactics))]], sim_matrix

            # Set DPP kernel to quality-diversity decomposition
            # quality is given by tactic probabilites
            vec_matrix = torch.mul(vec_matrix, probs.unsqueeze(1).to(self.device)).cpu().numpy()
            vec_matrix = vec_matrix @ vec_matrix.T

            DPP = FiniteDPP('likelihood', **{'L': vec_matrix})

            # rng = np.random.RandomState(1)
            try:
                DPP.sample_exact_k_dpp(size=num_filtered, mode='KuTa12')  # ,rng=rng)
            except Exception as e:
                logger.error(f"Error sampling from DPP: {e}")
                return [[i for i in range(len(tactics))]], sim_matrix

        return DPP.list_of_samples, sim_matrix
