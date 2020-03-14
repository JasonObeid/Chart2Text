#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: python dataOld/vocab.txt dataOld/train.txt
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import numpy as np
import sys
import argparse
import torch

from src.data.dictionary import Dictionary

def print_args(args):
    print("summary:\t{}".format(args.summary))
    print("summary_label:\t{}".format(args.summary_label))
    print("summary_vocab:\t{}".format(args.summary_vocab))
    print("summary_max_length:\t{}".format(args.summary_max_length))

if __name__ == '__main__':
    readme = ""
    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument('--summary', help = "summary dataOld")
    parser.add_argument('--summary_vocab', help = "summary dataOld vocab")
    parser.add_argument('--summary_label', help = "summary dataOld label")
    parser.add_argument('--summary_max_length', type=int, default=600, help = "summmary maximum length")
    args = parser.parse_args()

    if args.summary_vocab is None:
        args.summary_vocab = args.summary + "_vocab"
    if args.summary_label is None:
        args.summary_label = args.summary + "_label"

    assert os.path.isfile(args.summary)
    assert os.path.isfile(args.summary_vocab)
    assert os.path.isfile(args.summary_label)

    print_args(args)

    summary_dico = Dictionary.read_vocab(args.summary_vocab)
    summary_data = Dictionary.index_summary(args.summary, args.summary_label, summary_dico, 
                                            args.summary+".pth", max_len=args.summary_max_length)


