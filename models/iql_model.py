import copy
import math
from typing import Callable, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

select_batch_idxs = lambda x, idxs: torch.gather(x, dim=0, index=idxs.repeat(*x.shape[1:], 1).permute(len(x.shape) - 1,
                                                                                                      *list(range(
                                                                                                          len(x.shape) - 1))))
map_all_kvs = lambda f, kvs: tuple([tuple(map(f, items)) for items in kvs])

map_decoder_kvs = lambda f, kvs: tuple([tuple(map(f, items[:2])) + tuple(items[2:]) for items in kvs])

pad_sequence = lambda seq, to_len, val, device, dim: torch.cat(
    (seq, torch.full((*seq.shape[:dim], to_len - seq.shape[dim], *seq.shape[(dim + 1):]), val).to(device)), dim=dim)


def update_kvs(kvs, updated_kvs, lens_chosen, idx):
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs


def top_k_logits(logits, k):
    # logits = (batch, time, dim)
    _, bottom_k_idx = torch.topk(-logits, logits.shape[2] - k, dim=2)
    return torch.scatter(logits, dim=2, index=bottom_k_idx, value=float('-inf'))


def top_p_logits(logits, p):
    # logits = (batch, time, dim)
    sorted_logits, _ = torch.sort(logits, dim=2, descending=True)
    num_to_take = torch.sum(torch.cumsum(F.softmax(sorted_logits, dim=2), dim=2) <= p, dim=2).unsqueeze(2)
    mask = logits < torch.gather(sorted_logits, dim=2, index=torch.clamp(num_to_take, max=logits.shape[2] - 1))
    return logits.masked_fill(mask, float('-inf'))


def process_logits(logits, temp=1.0, top_k=None, top_p=None):
    logits /= temp
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    return logits


def always_terminate(s: np.ndarray):
    return True


def parameter_norm(model: nn.Module):
    norm = 0.0
    for param in model.parameters():
        norm += (param.norm() ** 2).item()
    return math.sqrt(norm)


def get_transformer_logs(attentions: List[torch.Tensor], model: nn.Module, attn_mask: torch.Tensor):
    logs = {}
    n = attn_mask.sum()

    model_attention_entropy = -sum(
        map(lambda x: ((x * torch.log(x + 1e-7)).sum(dim=-1) * attn_mask.unsqueeze(1)).sum().item(), attentions)) / (
                                      len(attentions) * n)
    model_parameter_norm = parameter_norm(model)

    logs['attention_entropy'] = (model_attention_entropy, n * len(attentions))
    logs['parameter_norm'] = (model_parameter_norm, 1)

    return logs


class TransformerMLP(nn.Module):
    def __init__(self, emb_dim, h_dim, out_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.ff1 = nn.Linear(emb_dim, h_dim)
        self.ff2 = nn.Linear(h_dim, emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.output_layer = nn.Linear(emb_dim, out_dim)

    def forward(self, x):
        return self.output_layer(
            self.ln2(x + F.dropout(self.ff2(F.gelu(self.ff1(self.ln1(x)))), p=self.dropout, training=self.training)))


class PerTokenIQL(nn.Module):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer,
                 alpha: float = 0.005,
                 gamma=1.0,
                 beta=1.0,
                 transition_weight=0.0,
                 clip_weight: Optional[float] = None,
                 value_max: Optional[float] = None,
                 value_min: Optional[float] = None,
                 detach_v: bool = False,
                 detach_pi: bool = False,
                 detach_q: bool = False,
                 double_q: bool = True,
                 tau: float = 0.9,
                 seperate_policy: bool = True,
                 seperate_target: bool = True,
                 exp_weights: bool = True,
                 advanced_mlp: bool = False,
                 cql_temp: float = 1.0,
                 max_len: int = 2048,
                 ):

        super().__init__()

        self.device = 'cuda'

        self.model = model

        self.max_len = max_len

        self.h_dim = self.model.config.d_model

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.transition_weight = transition_weight
        self.clip_weight = clip_weight
        self.value_max = value_max
        self.value_min = value_min
        self.detach_v = detach_v
        self.detach_pi = detach_pi
        self.detach_q = detach_q
        self.double_q = double_q
        self.tau = tau
        self.seperate_policy = seperate_policy
        self.seperate_target = seperate_target
        self.exp_weights = exp_weights
        self.advanced_mlp = advanced_mlp
        self.cql_temp = cql_temp
        self.tokenizer = tokenizer

        if not self.advanced_mlp:
            self.v = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, 1),
            )
        else:
            self.v = TransformerMLP(self.h_dim,
                                    4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner,
                                    1, self.model.config.resid_pdrop)
        if not self.advanced_mlp:
            self.q = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, self.tokenizer.vocab_size),
            )
        else:
            self.q = TransformerMLP(self.h_dim,
                                    4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner,
                                    self.tokenizer.vocab_size, self.model.config.resid_pdrop)
        if self.double_q:
            if not self.advanced_mlp:
                self.q2 = nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.h_dim * 2, self.tokenizer.vocab_size),
                )
            else:
                self.q2 = TransformerMLP(self.h_dim,
                                         4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner,
                                         self.tokenizer.vocab_size,
                                         self.model.config.resid_pdrop)
        if not self.advanced_mlp:
            self.target_q = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, self.tokenizer.vocab_size),
            )
        else:
            self.target_q = TransformerMLP(self.h_dim,
                                           4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner,
                                           self.tokenizer.vocab_size,
                                           self.model.config.resid_pdrop)
        if self.double_q:
            if not self.advanced_mlp:
                self.target_q2 = nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim * 2),
                    nn.ReLU(),
                    nn.Linear(self.h_dim * 2, self.tokenizer.vocab_size),
                )
            else:
                self.target_q2 = TransformerMLP(self.h_dim,
                                                4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner,
                                                self.tokenizer.vocab_size,
                                                self.model.config.resid_pdrop)

        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(local_param.data)

        if self.double_q:
            for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(local_param.data)

        if self.seperate_target:
            self.lm_target = copy.deepcopy(self.model)
        else:
            self.lm_target = None

        if self.seperate_policy:
            self.lm_policy = copy.deepcopy(self.model)
        else:
            self.lm_policy = None

        if self.lm_policy is None:
            self.pi = self.model.lm_head
        else:
            self.pi = self.lm_policy.lm_head

    def clip_values(self, values):
        if self.value_min is not None or self.value_max is not None:
            return torch.clip(values, self.value_min, self.value_max)
        return values

    def forward(self,
                input_ids: torch.Tensor,
                input_attn_mask: torch.Tensor,
                target_ids: torch.Tensor,
                target_attn_mask: torch.Tensor,
                qv_kwargs=None, policy_kwargs=None, target_kwargs=None,
                skip_policy_on_train=False,
                detach_full_policy=False
                ):

        if qv_kwargs is None:
            qv_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if policy_kwargs is None:
            policy_kwargs = {}

        if self.lm_target is None:
            qv_kwargs.update(target_kwargs)
        if self.lm_policy is None:
            qv_kwargs.update(policy_kwargs)

        input_attn_mask = input_attn_mask

        model_outputs = self.model(input_ids=input_ids,
                                   attention_mask=input_attn_mask,
                                   decoder_input_ids=target_ids,
                                   decoder_attention_mask=target_attn_mask,
                                   output_hidden_states=True,
                                   **qv_kwargs)

        all_model_outputs = {
            'qv_model_outputs': model_outputs,
            'policy_model_outputs': model_outputs,
            'target_model_outputs': model_outputs
        }

        if self.advanced_mlp:
            hidden_states = model_outputs.decoder_hidden_states[-2]
        else:
            hidden_states = model_outputs.decoder_hidden_states[-1]
        if self.lm_target is None:
            target_hidden_states = hidden_states
        else:
            with torch.no_grad():
                target_outputs = self.model(input_ids=input_ids,
                                            attention_mask=input_attn_mask,
                                            decoder_input_ids=target_ids,
                                            decoder_attention_mask=target_attn_mask,
                                            output_hidden_states=True,
                                            **qv_kwargs)

            all_model_outputs['target_model_outputs'] = target_outputs

            if self.advanced_mlp:
                target_hidden_states = target_outputs.decoder_hidden_states[-2]
            else:
                target_hidden_states = target_outputs.decoder_hidden_states[-1]

        if self.lm_policy is None:
            policy_hidden_states = model_outputs.decoder_hidden_states[-1]
        else:
            if skip_policy_on_train and self.training:
                policy_hidden_states = hidden_states
            else:
                if detach_full_policy:
                    with torch.no_grad():
                        policy_outputs = self.model(input_ids=input_ids,
                                                    attention_mask=input_attn_mask,
                                                    decoder_input_ids=target_ids,
                                                    decoder_attention_mask=target_attn_mask,
                                                    output_hidden_states=True,
                                                    **qv_kwargs)
                else:
                    policy_outputs = self.model(input_ids=input_ids,
                                                attention_mask=input_attn_mask,
                                                decoder_input_ids=target_ids,
                                                decoder_attention_mask=target_attn_mask,
                                                output_hidden_states=True,
                                                **qv_kwargs)

                all_model_outputs['policy_model_outputs'] = policy_outputs

                policy_hidden_states = policy_outputs.decoder_hidden_states[-1]

        state_hidden_states = hidden_states

        # action hidden_states will just ignore the last token (as no valid Q value)
        action_hidden_states = hidden_states[:, :-1, :]

        action_target_hidden_states = target_hidden_states[:, :-1, :]

        vs = self.v(state_hidden_states.detach() if self.detach_v else state_hidden_states).squeeze(2)

        qs = self.q(action_hidden_states.detach() if self.detach_q else action_hidden_states)

        if self.double_q:
            qs2 = self.q2(action_hidden_states.detach() if self.detach_q else action_hidden_states)

        with torch.no_grad():
            target_qs = self.target_q(action_target_hidden_states)
            if self.double_q:
                target_qs2 = self.target_q2(action_target_hidden_states)

        if skip_policy_on_train and self.training and self.lm_policy is not None:
            logits = torch.zeros((policy_hidden_states.shape[0], policy_hidden_states.shape[1],
                                  self.tokenizer.vocab_size,)).to(self.device)
        else:
            if detach_full_policy:
                with torch.no_grad():
                    logits = self.pi(policy_hidden_states.detach() if self.detach_pi else policy_hidden_states)
            else:
                logits = self.pi(policy_hidden_states.detach() if self.detach_pi else policy_hidden_states)

        return {
            'model_outputs': all_model_outputs,
            'vs': vs,
            'target_vs': vs,
            'qs': (qs, qs2,) if self.double_q else qs,
            'target_qs': self.clip_values(torch.minimum(target_qs, target_qs2) if self.double_q else target_qs),
            'logits': logits,
        }

    def get_downstream_rs(self, rs, gamma):
        gamma_row = torch.cumprod(torch.full(rs.shape, gamma).to(self.device), dim=1)
        gamma_tensor = torch.triu(gamma_row.unsqueeze(1) / gamma_row.unsqueeze(2))
        return (gamma_tensor * rs.unsqueeze(1)).sum(dim=2)

    # retrieve token weightings from advantage estimates, with self.transition_weight for non-action tokens
    def get_weights(self,
                    tokens: torch.Tensor,
                    vs: torch.Tensor,
                    qs: torch.Tensor,
                    action_ids: torch.Tensor
                    ):

        weights = torch.full(tokens.shape, self.transition_weight).to(self.device)

        if self.exp_weights:
            w_values = torch.exp(self.beta * (qs - vs))
        else:
            adv_sign = ((qs - vs) > 0.0).float()
            w_values = self.beta * adv_sign + (1 - self.beta) * (1 - adv_sign)

        if action_ids.shape[1] == 0:
            n = torch.zeros((tokens.shape[0],)).long().to(self.device)
        else:
            n = torch.argmax(action_ids, dim=1) + 1

        for i in range(tokens.shape[0]):
            # updates weights[i] at inds action_ids[i, :n[i]] with w_values[i, :n[i]]
            # keeping non action_ids as transition_weight
            weights[i] = torch.scatter(weights[i], dim=0, index=action_ids[i, :n[i]], src=w_values[i, :n[i]])

        if self.clip_weight is not None:
            weights = torch.clip(weights, max=self.clip_weight)

        return weights

    # Advantage Weighted Actor Critic (AWAC, https://arxiv.org/pdf/2006.09359.pdf)
    # Used to train the behavioural model \pi_{\beta}
    def awac_loss(self, tokens, attn_mask, logits, w):
        w = w.detach()

        losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1),
                                 reduction='none')

        losses = losses.reshape(tokens.shape[0], tokens.shape[1] - 1)

        return (losses * w[:, :-1] * attn_mask[:, 1:]).sum() / attn_mask[:, 1:].sum()

    def get_v_loss(self, vs, target_qs):
        target_qs = target_qs.detach()

        return (((target_qs >= vs).int() * self.tau * (target_qs - vs) ** 2 + (target_qs < vs).int() * (
                1 - self.tau) * (target_qs - vs) ** 2)).sum() / max(vs.shape[0], 1.0)

    def get_q_loss(self, vns, qs, rs, gamma):
        vns = vns.detach()

        if self.double_q:
            q1, q2 = qs
            l1 = ((vns * gamma + rs - q1) ** 2).sum() / vns.shape[0]
            l2 = ((vns * gamma + rs - q2) ** 2).sum() / vns.shape[0]
            return l1 + l2

        return ((vns * gamma + rs - qs) ** 2).sum() / vns.shape[0]

    def get_cql_loss(self, qs, select_tokens):
        if self.double_q:
            q1, q2 = qs
            b, t, d = q1.shape

            return (F.cross_entropy(q1.reshape(-1, d) / self.cql_temp, select_tokens.reshape(-1), reduction='sum') + (
                F.cross_entropy(q2.reshape(-1, d) / self.cql_temp, select_tokens.reshape(-1), reduction='sum'))) / b

        b, t, d = qs.shape

        return F.cross_entropy(qs.reshape(-1, d) / self.cql_temp, select_tokens.reshape(-1), reduction='mean')

    def get_qvs(self, items,
                qv_kwargs=None, policy_kwargs=None, target_kwargs=None,
                **kwargs):

        prepared_inputs = items

        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        target, target_mask = prepared_inputs['target'], prepared_inputs['target_mask']

        rs = prepared_inputs['rewards']

        self_outputs = self(tokens, attn_mask, target, target_mask,
                            qv_kwargs, policy_kwargs, target_kwargs,
                            **kwargs)

        model_outputs, vs, qs = self_outputs['model_outputs'], self_outputs['vs'], self_outputs['qs']

        target_qs, logits = self_outputs['target_qs'], self_outputs['logits']

        # values for the V updates (ignoring last)
        vt = vs[:, :-1]
        # values for the Q updates (ignoring first)
        vtp1 = vs[:, 1:]

        # set action_ids as the non-masked target indices
        b, t = vt.shape

        action_ids = torch.arange(0, t, device=self.device).repeat(b)

        action_ids = action_ids * target_mask[:, :-1]

        select_tokens = torch.gather(target[:, 1:], dim=1, index=action_ids)

        cql_term = self.get_cql_loss(qs, select_tokens)

        if self.double_q:
            q1, q2 = qs
            q1 = torch.gather(q1, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            q2 = torch.gather(q2, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
            qs = (q1, q2,)
            # tok_seq = [self.tokenizer.decode([token]) for token in select_tokens[0].detach().cpu().tolist()]
            # max_q_seq = torch.max(q1, q2)[0, :].detach().cpu().tolist()
            # print(self.tokenizer.decode(tokens[0, :][:attn_mask[0, :].sum().long()].tolist(),
            #                             clean_up_tokenization_spaces=False))
            # print(list(zip(tok_seq, max_q_seq)))
            # print(rs)
        else:
            qs = torch.gather(qs, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)

        target_qs = torch.gather(target_qs, dim=2, index=action_ids.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            weights = self.get_weights(target, vt, target_qs, action_ids)

        return {
            # tokens used are the targets
            'tokens': target,
            'attn_mask': target_mask,
            'model_outputs': model_outputs,
            'vs': vt,
            'qs': qs,
            'vns': vtp1,
            'target_vs': vt,
            'target_qs': target_qs,
            'target_vns': vtp1,
            'rs': rs,
            'logits': logits,
            'weights': weights,
            'cql_term': cql_term,
        }

    def get_loss(self,
                 items,
                 awac_weight=1.0,
                 v_loss_weight=1.0,
                 q_loss_weight=1.0,
                 cql_loss_weight=0.25,  # 1e-4
                 mc_returns=False):

        get_qvs_outputs = self.get_qvs(items,
                                       qv_kwargs={'output_attentions': True},
                                       policy_kwargs={'output_attentions': True},
                                       target_kwargs={'output_attentions': True},
                                       skip_policy_on_train=(awac_weight == 0.0),
                                       )

        tokens, attn_mask, model_outputs = get_qvs_outputs['tokens'], get_qvs_outputs['attn_mask'], get_qvs_outputs[
            'model_outputs']

        vs, qs = get_qvs_outputs['vs'], get_qvs_outputs['qs']

        vns, target_qs, rs = get_qvs_outputs['vns'], get_qvs_outputs['target_qs'], get_qvs_outputs['rs']

        logits, weights = get_qvs_outputs['logits'], get_qvs_outputs['weights']

        rs_downstream = self.get_downstream_rs(rs, self.gamma)

        if mc_returns:
            v_loss = self.get_v_loss(vs, rs_downstream)
        else:
            v_loss = self.get_v_loss(vs, target_qs)

        q_loss = self.get_q_loss(vns, qs, rs, self.gamma)

        cql_loss = get_qvs_outputs['cql_term']

        token_loss = self.awac_loss(tokens, attn_mask, logits, weights)

        # self.log({'v_loss': v_loss,
        #           'q_loss': q_loss,
        #           'cql_loss': cql_loss,
        #           'token_loss': token_loss})

        loss = awac_weight * token_loss + v_loss_weight * v_loss + q_loss_weight * q_loss + cql_loss_weight * cql_loss

        return loss


class IQL_Policy:
    def __init__(self, iql_model: PerTokenIQL,
                 kind: str, **generation_kwargs) -> None:

        self.iql_model = iql_model
        assert kind in {'beam', 'sample'}
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.kls_all = []
        self.logprobs_all = []

    def beam_raw(self,
                 tokens: torch.Tensor,
                 attn_mask: torch.Tensor,
                 state_ids: torch.Tensor,
                 action_ids: torch.Tensor,
                 termination_condition: Callable[[np.ndarray], bool],
                 max_generation_len: Optional[int] = None, beam_width=1,
                 temp=1.0, top_k=None, top_p=None,
                 exp_adv=False, adv_weight=0.0, adv_clip=None,
                 include_logits=True, include_adv=True, ):

        tokenizer = self.iql_model.tokenizer
        max_length = self.iql_model.max_len

        if max_length is None:
            max_length = self.iql_model.model.config.n_positions

        max_length = min(max_length, self.iql_model.model.config.n_positions)
        device = self.iql_model.device
        bsize, vocab_size = tokens.shape[0], tokenizer.vocab_size

        n = bsize * beam_width

        if max_generation_len is None:
            max_generation_len = max_length + 1

        input_strs = [
            tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(),
                             clean_up_tokenization_spaces=False)
            for i in range(len(tokens))]

        model_outputs = self.iql_model(tokens,
                                       attn_mask,
                                       state_ids,
                                       action_ids,
                                       qv_kwargs={'use_cache': True},
                                       policy_kwargs={'use_cache': True},
                                       target_kwargs={'use_cache': True})['model_outputs']

        kvs = {'qv': model_outputs['qv_model_outputs'].past_key_values}

        if self.iql_model.lm_target is not None:
            kvs['target'] = model_outputs['target_model_outputs'].past_key_values
        if self.iql_model.lm_policy is not None:
            kvs['policy'] = model_outputs['policy_model_outputs'].past_key_values

        original_dialogue_lens = attn_mask.sum(dim=1)

        batch_indicator = torch.stack(beam_width * [torch.arange(0, bsize).to(device)], dim=1)

        tokens = pad_sequence(torch.repeat_interleave(tokens, beam_width, dim=0), max_length,
                              tokenizer.pad_token_id,
                              device, 1)

        dialogue_lens = torch.repeat_interleave(original_dialogue_lens, beam_width, dim=0)

        kvs['qv'] = map_all_kvs(
            lambda x: pad_sequence(torch.repeat_interleave(x, beam_width, dim=0), max_length, 0.0, device, 2),
            kvs['qv'])
        if 'target' in kvs:
            kvs['target'] = map_all_kvs(
                lambda x: pad_sequence(torch.repeat_interleave(x, beam_width, dim=0), max_length, 0.0, device, 2),
                kvs['target'])
        if 'policy' in kvs:
            kvs['policy'] = map_all_kvs(
                lambda x: pad_sequence(torch.repeat_interleave(x, beam_width, dim=0), max_length, 0.0, device, 2),
                kvs['policy'])

        curr_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        logit_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        termination_mask = torch.full((n,), 1).to(device)

        state_idxs_temp, action_idxs_temp = torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device), torch.zeros(
            (dialogue_lens.shape[0], 1,)).long().to(device)

        t = torch.min(dialogue_lens).int()
        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)

        while termination_mask.sum() > 0 and t < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)
            curr_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['qv'])
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs

            if 'target' in kvs:
                curr_target_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['target'])
            if 'policy' in kvs:
                curr_policy_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['policy'])

            # since we are using past_key_values, only need to provide a single token at a time
            # as we only want the Q/V values for the one token, state/action idxs gives us this
            iql_outputs = self.iql_model(curr_token, None, state_idxs_temp, action_idxs_temp,
                                         qv_kwargs={'use_cache': True, 'past_key_values': curr_kvs},
                                         policy_kwargs={'use_cache': True, 'past_key_values': curr_policy_kvs},
                                         target_kwargs={'use_cache': True, 'past_key_values': curr_target_kvs})

            model_outputs, logits = iql_outputs['model_outputs'], iql_outputs['logits']

            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)

            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[
                torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(
                t < dialogue_lens, 1e7)

            edited_logits = process_logits(logits.clone(), temp=temp, top_k=top_k, top_p=top_p)

            vs, qs = iql_outputs['target_vs'], iql_outputs['target_qs']
            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)
            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)

            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)

            adv_logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = adv_logits[
                torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(
                t < dialogue_lens, 1e7)

            full_logits = (edited_logits if include_logits else 0.0) + (
                adv_logits if include_adv else 0.0) + base_logits.unsqueeze(1).unsqueeze(2)

            scores = (torch.log(F.softmax(full_logits, dim=-1)).reshape(1, bsize, beam_width, -1).permute(3, 0, 1, 2)
                      + curr_scores).permute(1, 2, 3, 0).reshape(1, bsize, -1)  # (time, batch, k*vocab)

            scores[0, :, vocab_size:] = scores[0, :, vocab_size:].masked_fill_(
                (t == original_dialogue_lens).unsqueeze(1).repeat(1, scores.shape[2] - vocab_size), float('-inf'))

            curr_scores, top_k_ = torch.topk(scores[0, :, :], k=beam_width, dim=1)  # (batch, k), (batch, k)

            tokens = tokens[(batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1), :]

            logits = logits[(batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1), :, :]

            logit_scores += torch.gather(torch.log(F.softmax(logits, dim=-1)).squeeze(1), dim=1,
                                         index=(top_k_.reshape(-1) % vocab_size).unsqueeze(1)).squeeze(1).reshape(-1,
                                                                                                                  beam_width)
            tokens[:, t] = top_k_.reshape(-1) % vocab_size  # (batch*k,)
            fixed_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                          rounding_mode='trunc')).reshape(
                -1), :, :, :], model_outputs['qv_model_outputs'].past_key_values)
            kvs['qv'] = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                          rounding_mode='trunc')).reshape(
                -1), :, :, :], kvs['qv'])
            kvs['qv'] = update_kvs(kvs['qv'], fixed_kvs, torch.arange(0, n).to(device), t - 1)
            if 'target' in kvs:
                fixed_target_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                                     rounding_mode='trunc')).reshape(
                    -1), :, :, :], model_outputs['target_model_outputs'].past_key_values)
                kvs['target'] = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                                  rounding_mode='trunc')).reshape(
                    -1), :, :, :], kvs['target'])
                kvs['target'] = update_kvs(kvs['target'], fixed_target_kvs, torch.arange(0, n).to(device),
                                           t - 1)
            if 'policy' in kvs:
                fixed_policy_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                                     rounding_mode='trunc')).reshape(
                    -1), :, :, :], model_outputs['policy_model_outputs'].past_key_values)
                kvs['policy'] = map_all_kvs(lambda x: x[(batch_indicator * beam_width + torch.div(top_k_, vocab_size,
                                                                                                  rounding_mode='trunc')).reshape(
                    -1), :, :, :], kvs['policy'])
                kvs['policy'] = update_kvs(kvs['policy'], fixed_policy_kvs, torch.arange(0, n).to(device),
                                           t - 1)
            termination_mask = termination_mask[(batch_indicator * beam_width + (top_k_ // vocab_size)).reshape(-1)]

            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (
                            1 - int(termination_condition(tokenizer.decode(tokens[idx, :].tolist(),
                                                                           clean_up_tokenization_spaces=False))))
            t += 1

            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in
                       range(n)]
        processed_outputs = []

        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(beam_width):
                processed_str = output_strs[i * beam_width + x][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                                    :processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                                    :processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(input_strs, processed_outputs)), curr_scores, -logit_scores

    def sample_raw(self,
                   tokens: torch.Tensor,
                   attn_mask: torch.Tensor,
                   state_ids: torch.Tensor,
                   action_ids: torch.Tensor,
                   termination_condition: Callable[[np.ndarray], bool],
                   num_generations=1, max_generation_len=None,
                   temp=1.0, top_k=None, top_p=None,
                   exp_adv=False, adv_weight=0.0, adv_clip=None,
                   include_logits=True, include_adv=True,
                   rerank_log_prob_weight: float = 0.0,
                   rerank_advantage_weight: float = 0.0,
                   ):

        assert include_logits or include_adv

        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.config.n_positions
        max_length = min(max_length, self.iql_model.model.config.n_positions)

        device = self.iql_model.device
        bsize = tokens.shape[0]

        n = bsize * num_generations

        if max_generation_len is None:
            max_generation_len = max_length + 1

        input_strs = [
            tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(),
                             clean_up_tokenization_spaces=False)
            for i in range(len(tokens))]

        model_outputs = self.iql_model(tokens, attn_mask, state_ids,
                                       action_ids,
                                       qv_kwargs={'use_cache': True},
                                       policy_kwargs={'use_cache': True},
                                       target_kwargs={'use_cache': True})['model_outputs']

        kvs = {'qv': model_outputs['qv_model_outputs'].past_key_values}

        if self.iql_model.lm_target is not None:
            kvs['target'] = model_outputs['target_model_outputs'].past_key_values

        if self.iql_model.lm_policy is not None:
            kvs['policy'] = model_outputs['policy_model_outputs'].past_key_values

        dialogue_lens = attn_mask.sum(dim=1)

        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), max_length,
                              tokenizer.pad_token_id, device, 1)

        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)

        kvs['qv'] = map_all_kvs(
            lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2),
            kvs['qv'])

        if 'target' in kvs:
            kvs['target'] = map_all_kvs(
                lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2),
                kvs['target'])

        if 'policy' in kvs:
            kvs['policy'] = map_all_kvs(
                lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2),
                kvs['policy'])

        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        kls = torch.full((dialogue_lens.shape[0],),
                         math.log(num_generations) - ((num_generations - 1) / num_generations)).to(device)

        advantages = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)

        state_idxs_temp, action_idxs_temp = torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device), torch.zeros(
            (dialogue_lens.shape[0], 1,)).long().to(device)

        t = torch.min(dialogue_lens).int()

        base_logits = torch.full((dialogue_lens.shape[0],), 0.0).to(device)

        while termination_mask.sum() > 0 and t < max_length:
            curr_token = tokens[:, t - 1].unsqueeze(1)

            curr_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['qv'])
            curr_target_kvs, curr_policy_kvs = curr_kvs, curr_kvs

            if 'target' in kvs:
                curr_target_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['target'])

            if 'policy' in kvs:
                curr_policy_kvs = map_all_kvs(lambda x: x[:, :, :t - 1, :], kvs['policy'])

            # since we are using past_key_values, only need to provide a single token at a time
            # as we only want the Q/V values for the one token, state/action idxs gives us this
            iql_outputs = self.iql_model(curr_token, None, state_idxs_temp, action_idxs_temp,
                                         qv_kwargs={'use_cache': True, 'past_key_values': curr_kvs},
                                         policy_kwargs={'use_cache': True, 'past_key_values': curr_policy_kvs},
                                         target_kwargs={'use_cache': True, 'past_key_values': curr_target_kvs})

            model_outputs, logits = iql_outputs['model_outputs'], iql_outputs['logits']

            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)

            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[
                torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(
                t < dialogue_lens, 1e7)

            edited_logits = process_logits(logits.clone(), temp=temp, top_k=top_k, top_p=top_p)

            vs, qs = iql_outputs['target_vs'], iql_outputs['target_qs']

            if exp_adv:
                adv_logits = adv_weight * (qs - vs.unsqueeze(2))
            else:
                adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
                adv_logits = adv_weight * adv_sign + (1 - adv_weight) * (1 - adv_sign)
                adv_logits = torch.log(adv_logits)

            if adv_clip is not None:
                adv_logits = torch.clip(adv_logits, max=adv_clip)

            adv_logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)

            adv_logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = adv_logits[
                torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(
                t < dialogue_lens, 1e7)

            full_logits = (edited_logits if include_logits else 0.0) + (
                adv_logits if include_adv else 0.0) + base_logits.unsqueeze(1).unsqueeze(2)

            cat_dist = torch.distributions.categorical.Categorical(logits=full_logits[:, 0])
            original_cat_dist = torch.distributions.categorical.Categorical(logits=logits[:, 0])

            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            kls += (cat_dist.log_prob(new_tokens) - original_cat_dist.log_prob(new_tokens))
            qs_chosen = torch.gather(qs.squeeze(1), dim=1, index=new_tokens.unsqueeze(1)).squeeze(1)
            advantages += (qs_chosen - vs.squeeze(1))
            tokens[:, t] = new_tokens

            kvs['qv'] = update_kvs(kvs['qv'], model_outputs['qv_model_outputs'].past_key_values,
                                   torch.arange(0, n).to(device), t - 1)
            if 'target' in kvs:
                kvs['target'] = update_kvs(kvs['target'], model_outputs['target_model_outputs'].past_key_values,
                                           torch.arange(0, n).to(device), t - 1)
            if 'policy' in kvs:
                kvs['policy'] = update_kvs(kvs['policy'], model_outputs['policy_model_outputs'].past_key_values,
                                           torch.arange(0, n).to(device), t - 1)

            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (
                            1 - int(termination_condition(tokenizer.decode(tokens[idx, :].tolist(),
                                                                           clean_up_tokenization_spaces=False))))
            t += 1
            termination_mask *= ((t - dialogue_lens) < max_generation_len).int()

        scores = ((advantages * rerank_advantage_weight) + (log_probs * rerank_log_prob_weight)).reshape(-1,
                                                                                                         num_generations)
        order = torch.argsort(-scores, dim=1)
        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in
                       range(len(tokens))]
        processed_outputs = []

        for i in range(len(input_strs)):
            temp_outputs = []

            for x in range(num_generations):
                processed_str = output_strs[i * num_generations + order[i, x]][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[
                                    :processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[
                                    :processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)

            processed_outputs.append(temp_outputs)

        scores = torch.gather(scores, dim=1, index=order)

        log_probs = torch.gather(log_probs.reshape(-1, num_generations), dim=1, index=order)
        kls = torch.gather(kls.reshape(-1, num_generations), dim=1, index=order)

        return list(zip(input_strs, processed_outputs)), log_probs.reshape(-1, num_generations), kls

    def generate(self, items,
                 termination_condition: Callable[[np.ndarray], bool], **kwargs):

        prepared_inputs = self.iql_model.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        state_idxs, action_ids = prepared_inputs['state_idxs'], prepared_inputs['action_ids']

        if self.kind == 'beam':
            method = self.beam_raw
        elif self.kind == 'sample':
            method = self.sample_raw
        else:
            raise NotImplementedError

        generations, info, kls = method(tokens, attn_mask,
                                        state_idxs, action_ids,
                                        termination_condition,
                                        **kwargs)
        return generations, info, kls

    # def score(self,
    #           state_idxs: Optional[torch.Tensor],
    #           attn_mask: Optional[torch.Tensor],
    #           action_ids: Optional[torch.Tensor],
    #           qv_kwargs=None,
    #           policy_kwargs=None,
    #           target_kwargs=None,
    #           beta: float = 1.0,
    #           exp_weights: bool = False,
    #           clip_weight: Optional[float] = None,
    #           logit_temp: float = 1.0,
    #           logit_top_k: Optional[int] = None,
    #           logit_top_p: Optional[float] = None,
    #           include_logits: bool = False,
    #           include_advantage: bool = True,
    #           action_mask: Optional[torch.Tensor] = None):
    #
    #     trivial_value_query = False
    #
    #     self_outputs = self(state_idxs, attn_mask, action_ids,
    #                         qv_kwargs, policy_kwargs, target_kwargs)
    #
    #     model_outputs = self_outputs['model_outputs']
    #     weights = torch.zeros(self_outputs['logits'].shape).to(self.device)
    #
    #     if include_advantage:
    #         if action_mask is None:
    #             action_mask = torch.ones((state_idxs.shape[0],)).to(self.device)
    #         vs, qs = self_outputs['target_vs'], self_outputs['target_qs']
    #         if not trivial_value_query:
    #             vs = vs[:, :-1]
    #         if exp_weights:
    #             w_values = beta * (qs - vs.unsqueeze(2))
    #         else:
    #             adv_sign = ((qs - vs.unsqueeze(2)) > 0.0).float()
    #             w_values = beta * adv_sign + (1 - beta) * (1 - adv_sign)
    #             w_values = torch.log(w_values)
    #         if clip_weight is not None:
    #             w_values = torch.clip(w_values, max=clip_weight)
    #         n = torch.argmax(action_ids, dim=1) + 1
    #         for i in range(state_idxs.shape[0]):
    #             weights[i] += torch.scatter(weights[i], dim=0,
    #                                         index=action_ids[i, :n[i]].unsqueeze(1).repeat(1, weights.shape[2]),
    #                                         src=w_values[i, :n[i], :]) * action_mask[i]
    #
    #     if include_logits:
    #         logits = process_logits(self_outputs['logits'], temp=logit_temp, top_k=logit_top_k, top_p=logit_top_p)
    #         weights += torch.log(F.softmax(logits, dim=-1))
    #
    #     return weights, model_outputs

    # def get_scores(self,
    #                items,
    #                beta: float = 1.0,
    #                exp_weights: bool = False,
    #                clip_weight: Optional[float] = None,
    #                logit_temp: float = 1.0,
    #                logit_top_k: Optional[int] = None,
    #                logit_top_p: Optional[float] = None,
    #                include_logits: bool = False,
    #                include_advantage: bool = True) -> torch.Tensor:
    #
    #     prepared_inputs = items
    #     tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
    #     s_idx, a_idx = prepared_inputs['state_idxs'], prepared_inputs['action_ids']
    #
    #     return self.score(s_idx, attn_mask, a_idx,
    #                       beta=beta, exp_weights=exp_weights, clip_weight=clip_weight,
    #                       logit_temp=logit_temp, logit_top_k=logit_top_k,
    #                       logit_top_p=logit_top_p, include_logits=include_logits,
    #                       include_advantage=include_advantage, action_mask=None)[0]
    #
    # def initial_score(self,
    #                   items,
    #                   beta: float = 1.0,
    #                   exp_weights: bool = False,
    #                   clip_weight: Optional[float] = None,
    #                   logit_temp: float = 1.0,
    #                   logit_top_k: Optional[int] = None,
    #                   logit_top_p: Optional[float] = None,
    #                   include_logits: bool = False,
    #                   include_advantage: bool = True) -> Tuple[torch.Tensor, Any]:
    #
    #     prepared_inputs = items
    #
    #     tokens = prepared_inputs['tokens']
    #     is_state = ((tokens == self.dataset.tokenizer.bos_token_id).float() + (
    #             tokens == self.dataset.tokenizer.eoa_token_id).float()) > 0
    #
    #     is_action = ((tokens == self.dataset.tokenizer.boa_token_id).float() + (
    #             tokens == self.dataset.tokenizer.eos_token_id).float()) > 0
    #
    #     state_points = torch.where(is_state, torch.arange(tokens.shape[1]).unsqueeze(0).repeat(tokens.shape[0], 1).to(
    #         self.device), -1)
    #
    #     action_points = torch.where(is_action, torch.arange(tokens.shape[1]).unsqueeze(0).repeat(tokens.shape[0], 1).to(
    #         self.device), -1)
    #
    #     action_mask = (action_points.argmax(dim=1) >= state_points.argmax(dim=1)).float()
    #     scores, model_outputs = self.score(tokens, None, None, None,
    #                                        qv_kwargs={'use_cache': True},
    #                                        policy_kwargs={'use_cache': True},
    #                                        target_kwargs={'use_cache': True},
    #                                        beta=beta, exp_weights=exp_weights,
    #                                        clip_weight=clip_weight,
    #                                        logit_temp=logit_temp, logit_top_k=logit_top_k,
    #                                        logit_top_p=logit_top_p, include_logits=include_logits,
    #                                        include_advantage=include_advantage, action_mask=action_mask)
    #
    #     return scores[:, -1, :], (
    #         model_outputs['qv_model_outputs'].past_key_values,
    #         model_outputs['policy_model_outputs'].past_key_values,
    #         model_outputs['target_model_outputs'].past_key_values,
    #         action_mask,
    #     )
    #
    # def next_score(self,
    #                tokens: torch.Tensor,
    #                state: Any,
    #                beta: float = 1.0,
    #                exp_weights: bool = False,
    #                clip_weight: Optional[float] = None,
    #                logit_temp: float = 1.0,
    #                logit_top_k: Optional[int] = None,
    #                logit_top_p: Optional[float] = None,
    #                include_logits: bool = False,
    #                include_advantage: bool = True) -> Tuple[torch.Tensor, Any]:
    #
    #     qv_kvs, policy_kvs, target_kvs, action_mask = state
    #     action_mask *= (tokens != self.dataset.tokenizer.eoa_token_id).float()
    #     action_mask += (tokens == self.dataset.tokenizer.eos_token_id).float()
    #     action_mask = (action_mask > 0.0).float()
    #
    #     scores, model_outputs = self.score(tokens.unsqueeze(1), None, None, None,
    #                                        qv_kwargs={'use_cache': True,
    #                                                   'past_key_values': qv_kvs},
    #                                        policy_kwargs={'use_cache': True,
    #                                                       'past_key_values': policy_kvs},
    #                                        target_kwargs={'use_cache': True,
    #                                                       'past_key_values': target_kvs},
    #                                        beta=beta, exp_weights=exp_weights, clip_weight=clip_weight,
    #                                        logit_temp=logit_temp, logit_top_k=logit_top_k,
    #                                        logit_top_p=logit_top_p, include_logits=include_logits,
    #                                        include_advantage=include_advantage, action_mask=action_mask)
    #
    #     return scores.squeeze(1), (
    #         model_outputs['qv_model_outputs'].past_key_values,
    #         model_outputs['policy_model_outputs'].past_key_values,
    #         model_outputs['target_model_outputs'].past_key_values,
    #         action_mask,
    #     )

    # def soft_update(self):
    #     for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
    #         target_param.data.copy_(self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)
    #     if self.double_q:
    #         for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
    #             target_param.data.copy_(self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)
    #     if self.lm_target is not None:
    #         for target_param, local_param in zip(self.lm_target.parameters(), self.model.parameters()):
    #             target_param.data.copy_(self.alpha * local_param.data + (1.0 - self.alpha) * target_param.data)
    #
    # def hard_update(self):
    #     for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
    #         target_param.data.copy_(local_param.data)
    #     if self.double_q:
    #         for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
    #             target_param.data.copy_(local_param.data)
    #     if self.lm_target is not None:
    #         del self.lm_target
    #         self.lm_target = None
    #         self.lm_target = copy.deepcopy(self.model)
    #
    # logs = {}
    # prepared_inputs = items
    # a_idx = prepared_inputs['action_ids']
    #
    # transformer_logs = {'qv_transformer_logs': get_transformer_logs(model_outputs['qv_model_outputs'].attentions,
    #                                                                 self.model, attn_mask)}
    #
    # if self.lm_policy is not None and (not (self.training and awac_weight == 0.0)):
    #     transformer_logs['policy_transformer_logs'] = get_transformer_logs(
    #         model_outputs['policy_model_outputs'].attentions, self.lm_policy, attn_mask)
    #
    # if self.lm_target is not None:
    #     transformer_logs['target_transformer_logs'] = get_transformer_logs(
    #         model_outputs['target_model_outputs'].attentions, self.lm_target, attn_mask)
    #
    # n = (1 - terminals[:, :-1]).sum().item()

    # logs['token_loss'] = (token_loss.item(), n)
    # logs['v_loss'] = (v_loss.item(), n)
    # logs['q_loss'] = (q_loss.item(), n)
    # logs['cql_loss'] = (cql_loss.item(), n)

    # advantages = sum(
    #     [((target_qs[i] - vs[i])[:(1 - terminals[i, :-1]).sum().long().item()]).detach().cpu().tolist() for i in
    #      range(tokens.shape[0])], [])

    # if self.double_q:
    #     q1, q2 = qs
    #     logs['q1_avg'] = ((q1 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
    #     logs['q1_var'] = (((((q1 - logs['q1_avg'][0]) ** 2) * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
    #     logs['q2_avg'] = ((q2 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
    #     logs['q2_var'] = (((((q2 - logs['q2_avg'][0]) ** 2) * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
    # else:
    #     logs['q_avg'] = ((qs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
    #     logs['q_var'] = (((((qs - logs['q_avg'][0]) ** 2) * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
    #
    # logs['v_avg'] = ((vs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
    # logs['v_var'] = (((((vs - logs['v_avg'][0]) ** 2) * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
    #
    # act_weights = torch.gather(weights, dim=1, index=a_idx)
    #
    # logs['act_weight_avg'] = (((act_weights * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), n)
    # logs['transformer'] = transformer_logs
    #
    # postproc_f = lambda l: l.update({'loss': awac_weight * l['token_loss'] + q_loss_weight * l[
    #     'q_loss'] + v_loss_weight * l['v_loss'] + cql_loss_weight * l['cql_loss']})
    #
    # hist_f = lambda l: l.update({'advantage_hist': wandb.Histogram(advantages)})
    #
    # return loss, logs, [postproc_f, hist_f]
