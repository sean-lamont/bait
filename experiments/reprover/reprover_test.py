from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-retriever-tacgen-byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean3-retriever-tacgen-byt5-small")

state = "n : ℕ\n⊢ gcd n n = n"
retrieved_premises = [
  "def <a>nat.gcd</a> : nat → nat → nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
  "@[simp] theorem <a>nat.mod_self</a> (n : nat) : n % n = 0",
]
input = "\n\n".join(retrieved_premises + [state])
print("------ INPUT ------\n", input)
tokenized_input = tokenizer(input, return_tensors="pt", max_length=2300, truncation=True)

# Generate a single tactic.
tactic_ids = model.generate(tokenized_input.input_ids, max_length=1024)
tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
print("\n------ OUTPUT ------")
print(tactic, end="\n\n")

# Generate multiple tactics via beam search.
tactic_candidates_ids = model.generate(
    tokenized_input.input_ids,
    max_length=1024,
    num_beams=4,
    length_penalty=0.0,
    do_sample=False,
    num_return_sequences=4,
    early_stopping=False,
)
tactic_candidates = tokenizer.batch_decode(
    tactic_candidates_ids, skip_special_tokens=True
)
for tac in tactic_candidates:
    print(tac)