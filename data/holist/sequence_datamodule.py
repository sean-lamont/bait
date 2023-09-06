import logging
from experiments.holist.utilities.sexpression_graphs import SExpressionGraph
import os.path

import torch
import random
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tqdm import tqdm
import re

def tokenize_string(string):
    pattern = r'(\(|\)|\s)'
    tokens = re.split(pattern, string)
    tokens = [token for token in tokens if token.strip()]  # Remove empty tokens
    return tokens

def sexpression_to_polish(sexpression_text):
    sexpression = SExpressionGraph()
    sexpression.add_sexp(sexpression_text)
    out = []

    def process_node(node):
        if len(sexpression.get_children(node)) == 0:
            out.append(node)
        for i, child in enumerate(sexpression.get_children(node)):
            if i == 0:
                out.append(sexpression.to_text(child))
                continue
            process_node(sexpression.to_text(child))

    process_node(sexpression.to_text(sexpression.roots()[0]))
    return out

def collate_and_pad_sequence(data_list, max_seq_len=3000):
    x = torch.nn.utils.rnn.pad_sequence(data_list)
    x = x[:max_seq_len]
    mask = (x == 0).T
    mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)
    return x, mask

"""
Data module returning a sequence for vanilla Transformer encoders
"""

from pymongo import MongoClient

client = MongoClient()
db = client['holist']
expr_collection = db['expression_graphs']


class HOListSequenceModule(LightningDataModule):
    def __init__(self, dir, batch_size):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def load(self):
        self.vocab = torch.load(self.dir + 'vocab.pt')
        # self.expr_dict = torch.load(self.dir + 'expr_dict.pt')

        logging.info("Loading expressions..")
        if os.path.exists(self.dir + 'sequence_dict.pt'):
            self.sequence_dict = torch.load(self.dir + 'sequence_dict.pt')
        else:
            self.expr_list = [k["_id"] for k in tqdm(expr_collection.find({}))]

        self.train_data = torch.load(self.dir + 'train_data.pt')
        self.val_data = torch.load(self.dir + 'val_data.pt')
        self.thm_ls = torch.load(self.dir + 'train_thm_ls.pt')

    def setup(self, stage: str) -> None:
        if stage == "fit":
            logging.info("Loading data..")
            self.load()

            # add brackets to vocab for sequence model
            if '(' not in self.vocab:
                self.vocab['('] = len(self.vocab)

            if ')' not in self.vocab:
                self.vocab[')'] = len(self.vocab)


            if not os.path.exists(self.dir + 'sequence_dict.pt'):
                logging.info("Generating sequence dictionary..")
                # self.sequence_dict = {k: [self.vocab[tok] if tok in self.vocab else self.vocab['UNK']for tok in tokenize_string(k)]
                #                   for k in tqdm(self.expr_list)}

                self.sequence_dict = {k: [self.vocab[tok] if tok in self.vocab else self.vocab['UNK']for tok in sexpression_to_polish(k)]
                                      for k in tqdm(self.expr_list)}

                torch.save(self.sequence_dict, self.dir + 'sequence_dict_polished.pt')


            self.sequence_dict = {k: torch.LongTensor(v) for k,v in self.sequence_dict.items()}

            self.train_data = self.filter(self.train_data)
            self.val_data = self.filter(self.val_data)
            self.thms_ls = [d for d in self.thm_ls if d in self.sequence_dict]

    def filter(self, data):
        def process(d):
            if d['goal'] in self.sequence_dict:
                if len(d['thms']) == 0:
                    return d
                thms = [th for th in d['thms'] if th in self.sequence_dict]
                if len(thms) > 0:
                    d['thms'] = thms
                    return d
                else:
                    return None
            else:
                return None

        return [process(d) for d in tqdm(data) if process(d) is not None]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def gen_batch(self, batch):
        # todo filter negative sampling to be disjoint from positive samples

        # batch will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [self.sequence_dict[x['goal']] for x in batch]

        # select random positive sample
        # if no parameters set it as a single element with '1' mapping to special token for no parameters
        pos_thms = [self.sequence_dict[random.choice(x['thms'])] if len(x['thms']) > 0 else torch.LongTensor([1])
                    for x in batch]

        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # 15 random negative samples per goal
        neg_thms = [[self.sequence_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]

        goals = collate_and_pad_sequence(goals)
        pos_thms = collate_and_pad_sequence(pos_thms)
        neg_thms = [collate_and_pad_sequence(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    module = HOListSequenceModule(dir='/home/sean/Documents/phd/deepmath-light/deepmath/processed_train_data/',
                                  batch_size=16)
    module.setup("fit")
    #
    # loader = module.train_dataloader()
    # print (next(iter(loader)))
    # i = 0
    # for b in tqdm(loader):
    #     i += 1
    #
    # print (i)
    #
    #
