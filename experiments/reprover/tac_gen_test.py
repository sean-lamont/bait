import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pymongo import MongoClient
import random

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small")  # Or "lean3" -> "lean4"
model = AutoModelForSeq2SeqLM.from_pretrained("kaiyuy/leandojo-lean3-tacgen-byt5-small",
                                              torch_dtype=torch.bfloat16).cuda()
# tokenizer.save_pretrained('./test_tok')
# model.save_pretrained('./test_model')
model = torch.compile(model)

states = [g['goal'] for g in MongoClient()['lean_dojo']['goal_data'].find()]

# random.shuffle(states)

beam_times = []
sample_times = []
sample_lengths = []

with torch.no_grad():
    for state in states[:100]:
        tokenized_state = tokenizer(state, return_tensors="pt")

        # Generate a single tactic.
        # tactic_ids = model.generate(tokenized_state.input_ids, max_length=1024)
        # tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
        # print(tactic, end="\n\n")
        #

        # num_samples = 64
        #
        # t0 = time.monotonic()
        #
        # output = model.generate(
        #     input_ids=tokenized_state.input_ids.cuda(),
        #     max_length=1024,
        #     do_sample=False,
        #     num_beams=num_samples,
        #     num_return_sequences=num_samples,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        #      length_penalty=0.0,
        #      early_stop=False
        # )

        # Return the output.
        # raw_output_text = tokenizer.batch_decode(
        #     output.sequences, skip_special_tokens=True
        # )
        #
        # transitions = model.compute_transition_scores(output.sequences, output.scores,
        #                                               normalize_logits=True)
        # output_text = []
        # output_score = []
        # for j in range(num_samples):
        #     t = raw_output_text[j]
        #     if t not in output_text:
        #         output_text.append(t)
        #         score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
        #         output_score.append(score)

        # elapsed = time.monotonic() - t0
        # beam_times.append(elapsed)

        # print(f'beam elapsed: {elapsed}')

        # for i in range(len(output_text)):
        #     print (f'Output text, score{output_text[i], output_score[i]}')

        output_text = []
        output_score = []

        t1 = time.monotonic()
        gen_ind = 0
        sample_len = 64
        while len(output_text) < 64:
            t0 = time.monotonic()
            gen_ind += 1
            output = model.generate(
                input_ids=tokenized_state.input_ids.cuda(),
                max_length=1024,
                do_sample=True,
                num_return_sequences=64,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Return the output.
            raw_output_text = tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )

            transitions = model.compute_transition_scores(output.sequences, output.scores,
                                                          normalize_logits=True)

            sample_len = 0
            for j in range(len(raw_output_text)):
                t = raw_output_text[j]
                if t not in output_text:
                    sample_len += 1
                    output_text.append(t)
                    score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
                    output_score.append(score)

            sample_lengths.append(sample_len)
            print(sample_len, time.monotonic() - t0)

        # for i in range(len(output_text)):
        #     print (f'Output text, score{output_text[i], output_score[i]}')

        elapsed = time.monotonic() - t1
        sample_times.append(elapsed)

        print(f'sample elapsed: {elapsed}, {len(output_text)} sequences\n')

    print(sum(sample_times) / len(sample_times))

        # print (len(output_text))
