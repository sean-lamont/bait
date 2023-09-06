import itertools
import random

import torch

from utils.mongodb_utils import get_batches


class CursorIter(torch.utils.data.IterableDataset):
    def __init__(self, cursor, fields, buf_size=4096):
        super(CursorIter).__init__()
        self.cursor = cursor
        self.batches = get_batches(self.cursor, batch_size=buf_size)
        self.curr_batches = next(self.batches)
        self.remaining = len(self.curr_batches)
        self.fields = fields

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            self.curr_batches = next(self.batches)
            random.shuffle(self.curr_batches)
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            ret = self.curr_batches.pop()
            return {field: ret[field] for field in self.fields}


class MongoStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, cursor, len, fields, buf_size=4096):
        super(MongoStreamDataset).__init__()
        self.ds = itertools.cycle(CursorIter(cursor, fields=fields, buf_size=buf_size))
        self.length = len

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.ds)
