from pymongo import MongoClient
import json

from tqdm import tqdm

if __name__ == '__main__':
    studentsList = []

    with open("processed/next_lemma_prediction.json") as f:
        lemma_data = [json.loads(obj) for obj in f]
    with open("processed/premise_classification.json") as f:
        premise_selection_data = [json.loads(obj) for obj in f]
    with open("processed/proof_step_classification.json") as f:
        proof_step_data = [json.loads(obj) for obj in f]
    with open("processed/proof_term_elab.json") as f:
        proof_term_data = [json.loads(obj) for obj in f]
    with open("processed/proof_term_prediction.json") as f:
        proof_term_pred_data = [json.loads(obj) for obj in f]
    with open("processed/result_elab.json") as f:
        result_elab_data = [json.loads(obj) for obj in f]
    with open("processed/skip_proof.json") as f:
        skip_proof_data = [json.loads(obj) for obj in f]
    with open("processed/theorem_name_prediction.json") as f:
        name_pred_data = [json.loads(obj) for obj in f]
    with open("processed/ts_elab.json") as f:
        ts_elab_data = [json.loads(obj) for obj in f]
    with open("processed/type_prediction.json") as f:
        type_pred_data = [json.loads(obj) for obj in f]

    db = MongoClient()
    db = db['leanstep']
    col = db['next_lemma_data']
    for x in tqdm(lemma_data):
        col.insert_one({"goal": x['goal'], "next_lemma": x['next_lemma']})

