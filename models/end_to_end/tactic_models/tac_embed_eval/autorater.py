import os
import time
import google.generativeai as genai

from transformers.utils import ModelOutput
from tqdm import tqdm
from experiments.end_to_end.stream_dataset import GoalStreamDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json

from pymongo import MongoClient

from models.end_to_end.tactic_models.tac_embed_separate.model import TransitionModel as SeparateModel
from models.end_to_end.tactic_models.tac_embed_large.model import TransitionModelLarge as CombinedModel


# load from data module
def collate_fn(examples):
    goal = [ex["tactic"] + ex["theorem"] + '\n\n' + ex["goal"] for ex in examples]

    tokenized_goal = tokenizer(
        goal,
        padding="longest",
        max_length=int(3000),
        truncation=True,
        return_tensors="pt",
    )

    result = [ex["result"] for ex in examples]

    tokenized_result = tokenizer(
        result,
        padding="longest",
        max_length=2000,
        truncation=True,
        return_tensors="pt",
    )

    tactic = [ex["tactic"] for ex in examples]

    tokenized_tactic = tokenizer(
        tactic,
        padding="longest",
        max_length=2000,
        truncation=True,
        return_tensors="pt",
    )

    lens = tokenized_tactic.attention_mask.sum(dim=1)

    result_ids = tokenized_result.input_ids
    result_ids[result_ids == tokenizer.pad_token_id] = -100

    tactic_ids = tokenized_tactic.input_ids

    batch = {}
    batch["goal"] = goal
    batch["goal_ids"] = tokenized_goal.input_ids
    batch["goal_mask"] = tokenized_goal.attention_mask
    batch["result"] = result
    batch["result_ids"] = tokenized_result.input_ids
    batch["result_mask"] = tokenized_goal.attention_mask
    batch["tactic"] = tactic
    batch["tactic_lens"] = lens
    batch["tactic_ids"] = tactic_ids
    batch["tactic_mask"] = tokenized_tactic.attention_mask

    return batch


def get_combined_enc(batch):
    goal_ids = batch["goal_ids"].cuda()
    goal_mask = batch["goal_mask"].cuda()
    tactic_lens = batch["tactic_lens"].cuda()

    full_enc = combined_model.get_full_encoding(goal_ids, goal_mask, tactic_lens)

    enc_outs = ModelOutput(last_hidden_state=full_enc)

    output = combined_model.decoder.generate(encoder_outputs=enc_outs,
                                             max_length=2000,
                                             num_beams=1,
                                             do_sample=False,
                                             num_return_sequences=1,
                                             early_stopping=True,
                                             output_scores=True,
                                             return_dict_in_generate=True,
                                             )

    # Return the output.

    output_text = tokenizer.batch_decode(
        output.sequences, skip_special_tokens=True
    )

    batch_size = goal_ids.size(0)

    num_samples = 1
    # for us, we only have one target (reference) so targets will be a list of lists,
    # with targets[i * num_samples: (i+1) * num_samples] being the target for the corresponding sample

    nl = '\n\n'

    # logger.info(f'Goal Before:\n {batch["goal"][0]}\n\n Goal After:\n  {batch["result"][0]} \n\n Predicted: \n{nl.join([o for o in output_text])}\n\n\n,')

    data = [[batch['goal'][i], batch['tactic'][i], batch['result'][i],
             nl.join(output_text[i * num_samples: (i + 1) * num_samples])]
            for i in range(batch_size)]

    return data


def get_separate_enc(batch):
    goal_ids = batch["goal_ids"].cuda()
    goal_mask = batch["goal_mask"].cuda()
    tactic_ids = batch["tactic_ids"].cuda()
    tactic_mask = batch["tactic_mask"].cuda()

    full_enc = separate_model.get_full_encoding(goal_ids, goal_mask, tactic_ids, tactic_mask)

    enc_outs = ModelOutput(last_hidden_state=full_enc)

    output = separate_model.decoder.generate(encoder_outputs=enc_outs,
                                             max_length=2000,
                                             num_beams=1,
                                             do_sample=False,
                                             num_return_sequences=1,
                                             early_stopping=True,
                                             output_scores=True,
                                             return_dict_in_generate=True,
                                             )

    # Return the output.
    output_text = tokenizer.batch_decode(
        output.sequences, skip_special_tokens=True
    )

    batch_size = goal_ids.size(0)

    nl = '\n\n'
    # logger.info(f'Goal Before:\n {batch["goal"][0]}\n\n Goal After:\n  {batch["result"][0]} \n\n Predicted: \n{nl.join([o for o in output_text])}\n\n\n,')

    num_samples = 1

    data = [[batch['goal'][i], batch['tactic'][i], batch['result'][i],
             nl.join(output_text[i * num_samples: (i + 1) * num_samples])]
            for i in range(batch_size)]

    return data


# %%
def eval_preds(batch, prompt):
    separate_preds = get_separate_enc(batch)
    combined_preds = get_combined_enc(
        batch
    )

    goal = '\n\n'.join(combined_preds[0][0].split('\n\n')[1:])
    tactic = combined_preds[0][1]
    result = combined_preds[0][2]

    combined_prediction = combined_preds[0][3]
    separate_prediction = separate_preds[0][3]

    prompt1 = prompt + f'Input: \n\nGoal: {goal}\n\nTactic: {tactic}\n\nTrue Outcome: {result}\n\nPrediction 1: {combined_prediction}\n\nPrediction 2: {separate_prediction}\n\nOutput: \n\n'


    response1 = None
    while not response1:
        try:
            response1 = model.generate_content(prompt1)
        except:
            time.sleep(10)

    # run again but swap order of predictions

    prompt2 = prompt + f'Input: \n\nGoal: {goal}\n\nTactic: {tactic}\n\nTrue Outcome: {result}\n\nPrediction 1: {separate_prediction}\n\nPrediction 2: {combined_prediction}\n\nOutput: \n\n'

    response2 = None
    while not response2:
        try:
            response2 = model.generate_content(prompt2)
        except:
            time.sleep(10)

    data = {'goal': goal,
            'tactic': tactic,
            'true_outcome': result,
            'combined_prediction': combined_prediction,
            'separate_prediction': separate_prediction,
            'response1': response1.text,
            'response2': response2.text}

    return data


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('sean-lamont/leandojo-lean3-reprover-novel-premises')

    database = 'lean_vae'
    collection = 'transitions'
    replace = 'keep'
    host = 'localhost:27017'  # mongodb hos

    fields = ['goal', 'tactic', 'result', 'theorem']

    val_filter = [{'$match': {'split': 'val'}},
                  {'$sort': {'rand_idx': 1}}]

    ds_val = GoalStreamDataset(db=database,
                               col_name=collection,
                               fields=fields,
                               filter_=val_filter,
                               gpu_id=0,
                               num_gpus=1,
                               host=host,
                               start_idx=36
                               )

    loader = DataLoader(ds_val,
                        collate_fn=collate_fn,
                        batch_size=1,
                        pin_memory=True
                        )

    combined_encoder_path = 'runs/diversity/large-single-vec-2/2024_07_15/10_57/checkpoints/last.ckpt'
    separate_encoder_path = 'runs/separate_tac_encoder/checkpoints/last.ckpt'

    combined_model = CombinedModel.load(combined_encoder_path, 'cuda:0', True)
    separate_model = SeparateModel.load(separate_encoder_path, 'cuda:0', True)

    genai.configure(api_key=os.environ['G_API_KEY'])

    model = genai.GenerativeModel('gemini-1.5-pro')

    with open('models/end_to_end/tactic_models/tac_embed_eval/prompt.txt', 'r') as f:
        prompt = f.read()

    req_per_min = 2

    collection = MongoClient()['lean_vae']['autorater_last_v_last']

    reqs = 0

    for batch in tqdm(loader):
        if reqs >= req_per_min:
            time.sleep(65)
            reqs = 0

        try:
            reqs += 2
            response = eval_preds(batch, prompt)

            # append to responses.jsonl file
            with open('runs/tac_embed_eval/responses.jsonl', 'a') as f:
                f.write(json.dumps(response) + '\n')

            # add to mongodb
            collection.insert_one(response)

        except Exception as e:
            print(e)
            time.sleep(10)
