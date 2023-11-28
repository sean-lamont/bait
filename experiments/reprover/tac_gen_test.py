from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")       # Or "lean3" -> "lean4"
model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")   # Or "lean3" -> "lean4"


# tokenizer.save_pretrained('./test_tok')
# model.save_pretrained('./test_model')

state = "n : ℕ\n⊢ gcd n n = n"
tokenized_state = tokenizer(state, return_tensors="pt")

# Generate a single tactic.
# tactic_ids = model.generate(tokenized_state.input_ids, max_length=1024)
# tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
# print(tactic, end="\n\n")
#
# # Generate multiple tactics via beam search.
# tactic_candidates_ids = model.generate(
#     tokenized_state.input_ids,
#     max_length=1024,
#     num_beams=4,
#     length_penalty=0.0,
#     do_sample=False,
#     num_return_sequences=4,
#     early_stopping=False,
# )
# tactic_candidates = tokenizer.batch_decode(
#     tactic_candidates_ids, skip_special_tokens=True
# )
# for tac in tactic_candidates:
#     print(tac)

num_samples = 64

output = model.generate(
    input_ids=tokenized_state.input_ids,
    max_length=1024,
    do_sample=True,
    num_return_sequences=num_samples,
    output_scores=True,
    return_dict_in_generate=True,
)

# Return the output.
raw_output_text = tokenizer.batch_decode(
    output.sequences, skip_special_tokens=True
)

transitions = model.compute_transition_scores(output.sequences, output.scores,
                                              normalize_logits=True)
output_text = []
output_score = []
for j in range(num_samples):
    t = raw_output_text[j]
    if t not in output_text:
        output_text.append(t)
        score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
        output_score.append(score)

for i in range(len(output_text)):
    print (f'Output text, score{output_text[i], output_score[i]}')


print ('top-p: \n')
output = model.generate(
    input_ids=tokenized_state.input_ids,
    max_length=1024,
    do_sample=True,
    num_return_sequences=num_samples,
    output_scores=True,
    return_dict_in_generate=True,
    top_p=0.95,
)

# Return the output.
raw_output_text = tokenizer.batch_decode(
    output.sequences, skip_special_tokens=True
)

transitions = model.compute_transition_scores(output.sequences, output.scores,
                                              normalize_logits=True)
output_text = []
output_score = []
for j in range(num_samples):
    t = raw_output_text[j]
    if t not in output_text:
        output_text.append(t)
        score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
        output_score.append(score)

for i in range(len(output_text)):
    print (f'Output text, score{output_text[i], output_score[i]}')

# print (len(output_text))
