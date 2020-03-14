# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()

class Dataset(object):

    def __init__(self, positions, summaries, summary_labels, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.summaries = summaries 
        self.summary_labels = summary_labels
        self.positions = positions
        self.summary_lengths = self.positions[:, 1] - self.positions[:, 0]

        # check number of sentences
        assert len(self.positions) == (self.summaries == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.positions)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.positions) == (self.summaries[self.positions[:, 1]] == eos).sum()  # check sentences indices
        # assert self.summary_lengths.min() > 0                                     # check empty sentences

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        # sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.positions)
        indices = np.arange(len(self.positions))
        indices = indices[self.summary_lengths[indices] > 0]
        self.positions = self.positions[indices]
        self.summary_lengths = self.positions[:, 1] - self.positions[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.positions)
        indices = np.arange(len(self.positions))
        indices = indices[self.summary_lengths[indices] <= max_len]
        self.positions = self.positions[indices]
        self.summary_lengths = self.positions[:, 1] - self.positions[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.positions)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.positions = self.positions[a:b]
        self.summary_lengths = self.positions[:, 1] - self.positions[:, 0]

        # re-index
        min_pos = self.positions.min()
        max_pos = self.positions.max()
        self.positions -= min_pos
        self.summaries = self.summaries[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.positions[sentence_ids]
            summaries = self.batch_sentences([self.summaries[a:b] for a, b in pos])
            summary_labels = self.batch_sentences([self.summary_labels[a:b] for a, b in pos])
            yield (summaries, summary_labels, sentence_ids) if return_indices else (summaries, summary_labels)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.positions) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.positions)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        lengths = self.summary_lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(self.positions))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)

class ParallelDataset(Dataset):

    def __init__(self, table_positions, table_entities, table_types, 
                table_values, table_feats, table_labels,
                summary_positions, summaries, summary_labels, 
                params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.table_positions = table_positions
        self.table_entities = table_entities
        self.table_types = table_types
        self.table_values = table_values
        self.table_feats = table_feats
        self.table_labels = table_labels

        self.summary_positions = summary_positions
        self.summaries = summaries
        self.summary_labels = summary_labels

        self.table_lengths = self.table_positions[:, 1] - self.table_positions[:, 0]
        self.summary_lengths = self.summary_positions[:, 1] - self.summary_positions[:, 0]

        # check number of sentences
        assert len(self.table_positions) == (self.table_entities == self.eos_index).sum()
        assert len(self.summary_positions) == (self.summaries == self.eos_index).sum()

        self.remove_empty_sentences()
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.table_positions)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.table_positions) == len(self.summary_positions) > 0
        assert len(self.table_positions) == (self.table_entities[self.table_positions[:, 1]] == eos).sum()
        assert len(self.summary_positions) == (self.summaries[self.summary_positions[:, 1]] == eos).sum()
        assert eos <= self.summaries.min() < self.summaries.max()
        assert self.table_lengths.min() > 0
        assert self.summary_lengths.min() > 0

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.table_positions)
        indices = np.arange(len(self.table_positions))
        indices = indices[self.table_lengths[indices] > 0]
        indices = indices[self.summary_lengths[indices] > 0]
        self.table_positions = self.table_positions[indices]
        self.summary_positions = self.summary_positions[indices]
        self.table_lengths = self.table_positions[:, 1] - self.table_positions[:, 0]
        self.summary_lengths = self.summary_positions[:, 1] - self.summary_positions[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.table_positions)
        indices = np.arange(len(self.table_positions))
        indices = indices[self.table_lengths[indices] <= max_len]
        indices = indices[self.summary_lengths[indices] <= max_len]
        self.table_positions = self.table_positions[indices]
        self.summary_positions = self.summary_positions[indices]
        self.table_lengths = self.table_positions[:, 1] - self.table_positions[:, 0]
        self.summary_lengths = self.summary_positions[:, 1] - self.summary_positions[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.table_positions)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.table_positions = self.table_positions[a:b]
        self.summary_positions = self.summary_positions[a:b]
        self.table_lengths = self.table_positions[:, 1] - self.table_positions[:, 0]
        self.summary_lengths = self.summary_positions[:, 1] - self.summary_positions[:, 0]

        # re-index
        min_pos1 = self.table_positions.min()
        max_pos1 = self.table_positions.max()
        min_pos2 = self.summary_positions.min()
        max_pos2 = self.summary_positions.max()
        self.table_positions -= min_pos1
        self.summary_positions -= min_pos2
        self.table_entities = self.table_entities[min_pos1:max_pos1 + 1]
        self.table_types = self.table_types[min_pos1:max_pos1 + 1]
        self.table_values = self.table_values[min_pos1:max_pos1 + 1]
        self.table_feats = self.table_feats[min_pos1:max_pos1 + 1]
        self.table_labels = self.table_labels[min_pos1:max_pos1 + 1]
        self.summaries = self.summaries[min_pos2:max_pos2 + 1]
        self.summary_labels = self.summary_labels[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            table_pos = self.table_positions[sentence_ids]
            summary_pos = self.summary_positions[sentence_ids]

            table_entities = self.batch_sentences([self.table_entities[a:b] for a, b in table_pos])
            table_types = self.batch_sentences([self.table_types[a:b] for a, b in table_pos])
            table_values = self.batch_sentences([self.table_values[a:b] for a, b in table_pos])
            table_feats = self.batch_sentences([self.table_feats[a:b] for a, b in table_pos])
            table_labels = self.batch_sentences([self.table_labels[a:b] for a, b in table_pos])

            summaries = self.batch_sentences([self.summaries[a:b] for a, b in summary_pos])
            summary_labels = self.batch_sentences([self.summary_labels[a:b] for a, b in summary_pos])

            yield (table_entities, table_types, table_values,
                   table_feats, table_labels, summaries, summary_labels, sentence_ids) if return_indices \
                   else (table_entities, table_types, table_values,
                   table_feats, table_labels, summaries, summary_labels)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.table_positions) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.table_positions)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.table_lengths + self.summary_lengths + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.table_positions))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)

class TableDataset(Dataset):
    
    def __init__(self, positions, table_entities, table_types, table_values, table_feats, table_labels, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        self.tokens_per_batch = -1 # table has constant size

        self.table_entities = table_entities
        self.table_types = table_types
        self.table_values = table_values
        self.table_feats = table_feats
        self.table_labels = table_labels
        self.positions = positions
        self.lengths = self.positions[:, 1] - self.positions[:, 0]
        assert len(self.table_entities) == len(self.table_labels)
        assert len(self.positions) == (self.table_entities == self.eos_index).sum()
        #assert all([each_len == self.lengths[0] for each_len in self.lengths])

        self.remove_empty_sentences()
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.positions)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.positions) == (self.table_entities[self.positions[:, 1]] == eos).sum()
        assert self.lengths.min() > 0

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.positions)
        indices = np.arange(len(self.positions))
        indices = indices[self.lengths[indices] > 0]
        self.positions = self.positions[indices]
        self.lengths = self.positions[:, 1] - self.positions[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.positions)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.positions = self.positions[a:b]
        self.lengths = self.positions[:, 1] - self.positions[:, 0]

        # re-index
        min_pos = self.positions.min()
        max_pos = self.positions.max()
        self.positions -= min_pos
        self.table_entities = self.table_entities[min_pos:max_pos + 1]
        self.table_types = self.table_types[min_pos:max_pos + 1]
        self.table_values = self.table_values[min_pos:max_pos + 1]
        self.table_feats = self.table_feats[min_pos:max_pos + 1]
        self.table_labels = self.table_labels[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.positions[sentence_ids]
            table_entities = self.batch_sentences([self.table_entities[a:b] for a, b in pos])
            table_types = self.batch_sentences([self.table_types[a:b] for a, b in pos])
            table_values = self.batch_sentences([self.table_values[a:b] for a, b in pos])
            table_feats = self.batch_sentences([self.table_feats[a:b] for a, b in pos])
            table_labels = self.batch_sentences([self.table_labels[a:b] for a, b in pos])
            yield (table_entities, table_types, table_values, 
                    table_feats, table_labels, sentence_ids) if return_indices \
                    else (table_entities, table_types, table_values,
                    table_feats, table_labels)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.positions) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.positions)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths 

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.positions))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)

