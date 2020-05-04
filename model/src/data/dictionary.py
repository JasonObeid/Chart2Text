# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from logging import getLogger


logger = getLogger()


BOS_WORD = '<s>'
EOS_WORD = '</s>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10

SEP_WORD = SPECIAL_WORD % 0
MASK_WORD = SPECIAL_WORD % 1


class Dictionary(object):

    def __init__(self, id2word, word2id):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        assert all(self.id2word[4 + i] == SPECIAL_WORD % i for i in range(SPECIAL_WORDS))
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.unk_index)

    def max_vocab(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.check_valid()
        logger.info("Maximum vocabulary size: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (max_vocab, init_size, len(self), init_size - len(self)))

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i in range(SPECIAL_WORDS):
            word2id[SPECIAL_WORD % i] = 4 + i
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            fields = line.rstrip().split()
            if len(fields) <= 0 or len(fields) > 2:
                skipped += 1
                continue

            if fields[0] in word2id:
                skipped += 1
                print('%s already in vocab' % fields[0])
                continue
            word2id[fields[0]] = 4 + SPECIAL_WORDS + i - skipped  # shift because of extra words
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def index_data(path, bin_path, dico):
        """
        Index sentences with a dictionary.
        """
        if bin_path is not None and os.path.isfile(bin_path):
            print("Loading dataOld from %s ..." % bin_path)
            data = torch.load(bin_path)
            assert dico == data['dico']
            return data

        positions = []
        sentences = []
        unk_words = {}

        # index sentences
        f = open(path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(i)
            s = line.rstrip().split()
            # skip empty sentences
            if len(s) == 0:
                print("Empty sentence in line %i." % i)
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w, no_unk=False)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + SPECIAL_WORDS and word_id != 3:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(1)  # EOS index
        f.close()

        # tensorize dataOld
        positions = np.int64(positions)
        if len(dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception("Dictionary is too big.")

        assert sentences.min() >= 0
        data = {
            'dico': dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
        }
        if bin_path is not None:
            print("Saving the dataOld to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data

    @staticmethod
    def index_table(table_path, table_label_path, table_dico, bin_path):
        """
        """
        if bin_path is not None and os.path.isfile(bin_path):
            print("Loading dataOld from %s ..." % bin_path)
            data = torch.load(bin_path)
            assert table_dico == data['dico']
            print("%s checked. Nothing is done." % bin_path)
            return data

        positions = []
        table_seq_entity = []
        table_seq_type = []
        table_seq_value = []
        table_seq_feat = []
        table_seq_label = []

        # index table_contents
        table_inf = open(table_path, 'r', encoding='utf-8')
        table_label_inf = open(table_label_path, 'r', encoding='utf-8')

        for i, (line, label_line) in enumerate(zip(table_inf, table_label_inf)):
            table_items = line.strip().split()
            table_label = label_line.strip().split()
            assert len(table_items) == len(table_label)
            # skip empty table_contents
            if len(table_items) == 0:
                print("Empty sentence in line %i." % i)
                continue

            # entity, type, value, feat
            table_entity_indexed = []
            table_type_indexed = []
            table_value_indexed = []
            table_feat_indexed = []

            table_label_indexed = []

            for item, label in zip(table_items, table_label):
                fields = item.split('|')
                assert len(fields) == 4
                entity_id = table_dico.index(fields[0], no_unk=False)
                type_id = table_dico.index(fields[1], no_unk=False)
                value_id = table_dico.index(fields[2], no_unk=False)
                feat_id = table_dico.index(fields[3], no_unk=False)

                label_id = int(label)

                table_entity_indexed.append(entity_id)
                table_type_indexed.append(type_id)
                table_value_indexed.append(value_id)
                table_feat_indexed.append(feat_id)

                table_label_indexed.append(label_id)

            # add sentence
            positions.append([len(table_seq_entity), len(table_seq_entity) + len(table_entity_indexed)])
            table_seq_entity.extend(table_entity_indexed)
            table_seq_entity.append(1)  # EOS index

            table_seq_type.extend(table_type_indexed)
            table_seq_type.append(0)  # empty feat

            table_seq_value.extend(table_value_indexed)
            table_seq_value.append(0)

            table_seq_feat.extend(table_feat_indexed)
            table_seq_feat.append(0)

            table_seq_label.extend(table_label_indexed)
            table_seq_label.append(0)

        table_inf.close()
        table_label_inf.close()

        # tensorize dataOld
        positions = np.int64(positions)
        if len(table_dico) < 1 << 16:
            table_seq_entity = np.uint16(table_seq_entity)
            table_seq_type = np.uint16(table_seq_type)
            table_seq_value = np.uint16(table_seq_value)
            table_seq_feat = np.uint16(table_seq_feat)
        elif len(table_dico) < 1 << 31:
            table_seq_entity = np.int32(table_seq_entity)
            table_seq_type = np.uint32(table_seq_type)
            table_seq_value = np.uint32(table_seq_value)
            table_seq_feat = np.uint32(table_seq_feat)
        else:
            raise Exception("Dictionary is too big.")

        table_seq_label = np.uint8(table_seq_label)

        data = {
            'dico': table_dico,
            'positions': positions,
            'table_entities': table_seq_entity,
            'table_types': table_seq_type,
            'table_values': table_seq_value,
            'table_feats': table_seq_feat,
            'table_labels': table_seq_label,
        }
        if bin_path is not None:
            print("Saving the dataOld to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data

    @staticmethod
    def index_summary(summary_path, summary_label_path, dico, bin_path, max_len=600):
        """
        Index summaries with a dictionary.
        """
        if bin_path is not None and os.path.isfile(bin_path):
            print("Loading dataOld from %s ..." % bin_path)
            data = torch.load(bin_path)
            assert dico == data['dico']
            print("%s checked. Nothing is done." % bin_path)
            return data

        positions = []
        summaries = []
        summary_labels = []

        # index summaries
        summary_inf = open(summary_path, 'r', encoding='utf-8')
        summary_label_inf = open(summary_label_path, 'r', encoding='utf-8')

        for i, (summary_line, label_line) in enumerate(zip(summary_inf, summary_label_inf)):
            if i % 1000000 == 0 and i > 0:
                print(i)
            summary_tokens = summary_line.rstrip().split()
            summary_token_labels = label_line.rstrip().split()
            assert len(summary_tokens) == len(summary_token_labels)
            # skip empty summaries
            if len(summary_tokens) == 0:
                print("Empty sentence in line %i." % i)
                continue
            # index sentence words
            summary_indexed = []
            summary_label_indexed = []
            for token, label in zip(summary_tokens, summary_token_labels):
                word_id = dico.index(token, no_unk=False)
                label_id = int(label)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + SPECIAL_WORDS and word_id != 3:
                    logger.warning('Found unexpected special word "%s" (%i)!!' % (token, word_id))
                    continue
                assert word_id >= 0

                summary_indexed.append(word_id)
                summary_label_indexed.append(label_id)
            if len(summary_indexed) > max_len:
                summary_indexed = summary_indexed[:max_len]
                summary_label_indexed = summary_label_indexed[:max_len]

            # add sentence
            positions.append([len(summaries), len(summaries) + len(summary_indexed)])
            summaries.extend(summary_indexed)
            summaries.append(1)  # EOS index
            summary_labels.extend(summary_label_indexed)
            summary_labels.append(0)

        summary_inf.close()
        summary_label_inf.close()

        # tensorize dataOld
        positions = np.int64(positions)
        if len(dico) < 1 << 16:
            summaries = np.uint16(summaries)
        elif len(dico) < 1 << 31:
            summaries = np.int32(summaries)
        else:
            raise Exception("Dictionary is too big.")

        summary_labels = np.uint8(summary_labels)

        assert summaries.min() >= 0
        data = {
            'dico': dico,
            'positions': positions,
            'summaries': summaries,
            'summary_labels': summary_labels,
        }
        if bin_path is not None:
            print("Saving the dataOld to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data
