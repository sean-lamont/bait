import torch
from typing import Union, List
from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small")
model = T5EncoderModel.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small")

state = "n : ℕ\n⊢ gcd n n = n"
premises = [
  "<a>vsub_eq_zero_iff_eq</a> @[simp] lemma vsub_eq_zero_iff_eq {p1 p2 : P} : p1 -ᵥ p2 = (0 : G) ↔ p1 = p2",
  "<a>is_scalar_tower.coe_to_alg_hom'</a> @[simp] lemma coe_to_alg_hom' : (to_alg_hom R S A : S → A) = algebra_map S A",
  "<a>polynomial.X_sub_C_ne_zero</a> theorem X_sub_C_ne_zero (r : R) : X - C r ≠ 0",
  "<a>forall_true_iff</a> theorem forall_true_iff : (α → true) ↔ true",
  "def <a>nat.gcd</a> : nat → nat → nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>nat.gcd_zero_left</a> (x : nat) : gcd 0 x = x",
  "@[simp] theorem <a>nat.gcd_succ</a> (x y : nat) : gcd (succ x) y = gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>nat.mod_self</a> (n : nat) : n % n = 0",
]  # A corpus of premises to retrieve from.

@torch.no_grad()
def encode(s: Union[str, List[str]]) -> torch.Tensor:
    """Encode texts into feature vectors."""
    if isinstance(s, str):
        s = [s]
        should_squeeze = True
    else:
        should_squeeze = False
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    if should_squeeze:
      features = features.squeeze()
    return features

@torch.no_grad()
def retrieve(state: str, premises: List[str], k: int) -> List[str]:
    """Retrieve the top-k premises given a state."""
    state_emb = encode(state)
    premise_embs = encode(premises)
    scores = (state_emb @ premise_embs.T)
    topk = scores.topk(k).indices.tolist()
    return [premises[i] for i in topk]

for p in retrieve(state, premises, k=3):
    print(p, end="\n\n")