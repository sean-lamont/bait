'''
Generator util to iterate through a MongoDB cursor with a given batch size
'''
import itertools


def get_batches(cursor, batch_size):
    batch = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            yield batch
            del batch[:]
        batch.append(row)
    yield batch



def get_all_batches(cursor, batch_size):
    batch = []
    batches = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            # yield batch
            batches.append(batch)
            del batch[:]
        batch.append(row)
    return batches
