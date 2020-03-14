# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerEncoder, TransformerDecoder

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Summary generation")

    # main parameters
    parser.add_argument("--model_path", type=str, default="./model_training/", help="Experiment dump path")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--table_path", type=str, default="", help="table path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    parser.add_argument("--beam_size", type=int, default=1, help="beam size")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="")
    parser.add_argument("--early_stopping", type=bool, default=False, help="")

    return parser


def main(params):
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])

    # update dictionary parameters
    for name in ['src_n_words', 'tgt_n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    source_dico = Dictionary(reloaded['source_dico_id2word'], reloaded['source_dico_word2id'])
    target_dico = Dictionary(reloaded['target_dico_id2word'], reloaded['target_dico_word2id'])
    encoder = TransformerEncoder(model_params, source_dico, with_output=False).cuda().eval()
    decoder = TransformerDecoder(model_params, target_dico, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded['encoder'])
    decoder.load_state_dict(reloaded['decoder'])

    # read sentences from stdin
    table_lines = []
    table_inf = open(params.table_path, 'r', encoding='utf-8')

    for table_line in table_inf:
        table_lines.append(table_line)

    outf = io.open(params.output_path, 'w', encoding='utf-8')

    for i in range(0, len(table_lines), params.batch_size):
        # prepare batch
        enc_x1_ids = []
        enc_x2_ids = []
        enc_x3_ids = []
        enc_x4_ids = []
        for table_line in table_lines[i:i + params.batch_size]:
            #print(table_line)
            record_seq = [each.split('|') for each in table_line.split()]
            print(record_seq)
            for x in record_seq:
                print(x)
                print(' ')
            assert all([len(x) == 4 for x in record_seq])
            enc_x1_ids.append(torch.LongTensor([source_dico.index(x[0]) for x in record_seq]))
            enc_x2_ids.append(torch.LongTensor([source_dico.index(x[1]) for x in record_seq]))
            enc_x3_ids.append(torch.LongTensor([source_dico.index(x[2]) for x in record_seq]))
            enc_x4_ids.append(torch.LongTensor([source_dico.index(x[3]) for x in record_seq]))

        enc_xlen = torch.LongTensor([len(x) + 2 for x in enc_x1_ids])
        enc_x1 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(params.pad_index)
        enc_x1[0] = params.eos_index
        enc_x2 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(params.pad_index)
        enc_x2[0] = params.eos_index
        enc_x3 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(params.pad_index)
        enc_x3[0] = params.eos_index
        enc_x4 = torch.LongTensor(enc_xlen.max().item(), enc_xlen.size(0)).fill_(params.pad_index)
        enc_x4[0] = params.eos_index

        for j, (s1,s2,s3,s4) in enumerate(zip(enc_x1_ids, enc_x2_ids, enc_x3_ids, enc_x4_ids)):
            if enc_xlen[j] > 2:  # if sentence not empty
                enc_x1[1:enc_xlen[j] - 1, j].copy_(s1)
                enc_x2[1:enc_xlen[j] - 1, j].copy_(s2)
                enc_x3[1:enc_xlen[j] - 1, j].copy_(s3)
                enc_x4[1:enc_xlen[j] - 1, j].copy_(s4)
            enc_x1[enc_xlen[j] - 1, j] = params.eos_index
            enc_x2[enc_xlen[j] - 1, j] = params.eos_index
            enc_x3[enc_xlen[j] - 1, j] = params.eos_index
            enc_x4[enc_xlen[j] - 1, j] = params.eos_index

        enc_x1 = enc_x1.cuda()
        enc_x2 = enc_x2.cuda()
        enc_x3 = enc_x3.cuda()
        enc_x4 = enc_x4.cuda()
        enc_xlen = enc_xlen.cuda()

        # encode source batch and translate it
        encoder_output = encoder('fwd', x1=enc_x1, x2=enc_x2, x3=enc_x3, x4=enc_x4, lengths=enc_xlen)
        encoder_output = encoder_output.transpose(0, 1)

        # max_len = int(1.5 * enc_xlen.max().item() + 10)
        max_len = 602
        if params.beam_size <= 1:
            decoded, dec_lengths = decoder.generate(encoder_output, enc_xlen, max_len=max_len)
        elif params.beam_size > 1:
            decoded, dec_lengths = decoder.generate_beam(encoder_output, enc_xlen, params.beam_size, 
                                            params.length_penalty, params.early_stopping, max_len=max_len)

        for j in range(decoded.size(1)):

            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = table_lines[i + j].strip()
            target = " ".join([target_dico[sent[k].item()] for k in range(len(sent))])
            sys.stderr.write("%i / %i: %s\n" % (i + j, len(table_lines), target))
            outf.write(target + "\n")

    outf.close()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
#    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)


