import pickle

from tqdm import tqdm

from experiments.end_to_end.common import zip_strict
from models.end_to_end.tactic_models.retrieval.datamodule import RetrievalDataModule
from models.end_to_end.tactic_models.retrieval.model import PremiseRetriever

if __name__ == '__main__':
    datamodule = RetrievalDataModule(data_path='data/LeanDojo/data/leandojo_benchmark/novel_premises',
                                     # corpus_path='runs/indexed_corpus',
                                     corpus_path='data/LeanDojo/data/leandojo_benchmark/corpus.jsonl',
                                     model_name='sean-lamont/leandojo-lean3-reprover-novel-premises',
                                     num_negatives=3,
                                     num_in_file_negatives=1,
                                     batch_size=8,
                                     eval_batch_size=180,
                                     max_seq_len=1024,
                                     num_workers=0,
                                     )

    datamodule.setup()

    retriever = PremiseRetriever.load(
        'runs/retriever_novel_premises.ckpt', 'cuda', freeze=True
    )

    retriever.load_corpus('runs/indexed_corpus')

    predict_step_outputs = []

    corpus_embs = retriever.corpus_embeddings.cuda()

    for batch in tqdm(datamodule.predict_dataloader()):
        batch = datamodule.transfer_batch_to_device(batch=batch, device=retriever.device, dataloader_idx=0)
        context_emb = retriever._encode(batch["context_ids"], batch["context_mask"])
        assert not retriever.embeddings_staled


        retrieved_premises, scores = retriever.corpus.get_nearest_premises(
            corpus_embs,
            batch["context"],
            context_emb,
            retriever.num_retrieved,
        )

        for (
                url,
                commit,
                file_path,
                full_name,
                start,
                tactic_idx,
                ctx,
                pos_premises,
                premises,
                s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )


        path = "runs/predictions.pickle"
        with open(path, "wb") as oup:
            pickle.dump(predict_step_outputs, oup)
