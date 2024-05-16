from itertools import islice
import torch
from loguru import logger
from pymongo import MongoClient

from utils.utils import get_batches


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_info.id
    dataset.num_workers = worker_info.num_workers
    dataset.setup()


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
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            ret = self.curr_batches.pop()
            return {field: ret[field] for field in self.fields}


# todo reloading dataset every 38912??
class GoalStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 db,
                 col_name,
                 fields,
                 filter_,
                 gpu_id=0,
                 num_gpus=1,
                 worker_id=0,
                 num_workers=1,
                 buf_size=2048,
                 start_idx=0):
        super(GoalStreamDataset).__init__()

        self.ds = None
        self.db = db
        self.col_name = col_name
        self.worker_id = worker_id
        self.fields = fields
        self.buf_size = buf_size
        self.filter_ = filter_
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.start_idx = start_idx

        self.query = self.filter_ + [{'$project': {v: 1 for v in self.fields}},
                                     {'$skip': self.start_idx}]

        if '_id' not in self.fields:
            self.query[-2]['$project']['_id'] = 0

        collection = MongoClient()[self.db][self.col_name]

        # run through once to get the length of cursor
        length = list(collection.aggregate(
            self.filter_ + [{'$count': 'length'}]))[0][
            'length']

        self.length = length // num_gpus

        cursor = collection.aggregate(self.query)

        self.cursor_iter = CursorIter(cursor, fields=self.fields, buf_size=self.buf_size)

        self.setup()

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def reset(self, idx):
        self.__init__(self.db,
                      self.col_name,
                      self.fields,
                      self.filter_,
                      self.gpu_id,
                      self.num_gpus,
                      self.worker_id,
                      self.num_workers,
                      self.buf_size,
                      idx)

    def __next__(self):
        try:
            next_ = next(self.ds)
            self.start_idx += 1
            return next_
        except StopIteration:
            self.reset(0)
            return next(self.ds)
        except Exception as e:
            self.reset(self.start_idx)
            logger.warning(f'Loader exception {e}, reloading dataset {len(self)}..')
            return next(self.ds)

    def setup(self):
        total_workers = self.num_gpus * self.num_workers
        global_idx = (self.gpu_id * self.num_workers) + self.worker_id

        # make the dataset iterator return unique values for each worker, and ensure they all have the same number of
        # elements
        self.ds = islice(self.cursor_iter, global_idx, None, total_workers)
