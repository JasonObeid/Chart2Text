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

hv_keys = ['H', 'V']
wl_keys = ['W', 'L']
key_keys = ['AST', 'BLK', 'DREB', 'FG3A', 'FG3M', 'FG3_PCT', 'FGA', 'FGM', 'FG_PCT', 
            'FIRST_NAME', 'FTA', 'FTM', 'FT_PCT', 'MIN', 'OREB', 'PF', 'PLAYER_NAME', 
            'PTS', 'REB', 'SECOND_NAME', 'START_POSITION', 'STL', 'TEAM-AST', 
            'TEAM-CITY', 'TEAM-FG3_PCT', 'TEAM-FG_PCT', 'TEAM-FT_PCT', 'TEAM-LOSSES', 
            'TEAM-NAME', 'TEAM-PTS', 'TEAM-PTS_QTR1', 'TEAM-PTS_QTR2', 'TEAM-PTS_QTR3', 
            'TEAM-PTS_QTR4', 'TEAM-REB', 'TEAM-TOV', 'TEAM-WINS', 'TO']

def print_args(args):
    print("table:\t{}".format(args.table))
    print("table_label:\t{}".format(args.table_label))
    print("table_vocab:\t{}".format(args.table_vocab))

if __name__ == '__main__':
    readme = ""
    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument('--table', help = "table dataOld")
    parser.add_argument('--table_label', help = "table label")
    parser.add_argument('--table_vocab', help = "table vocab")
    args = parser.parse_args()

    if args.table_label is None:
        args.table_label = args.table + "_label"
    if args.table_vocab is None:
        args.table_vocab = args.table + "_vocab"

    assert os.path.isfile(args.table)
    assert os.path.isfile(args.table_label)
    assert os.path.isfile(args.table_vocab)

    print_args(args)

    table_dico = Dictionary.read_vocab(args.table_vocab)
    table_data = Dictionary.index_table(args.table, args.table_label, table_dico, args.table+".pth")

