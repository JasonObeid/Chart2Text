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
    # Generate a parameters parser.
    # parse parameters
    
    parser = argparse.ArgumentParser(description="Summary generation")

    # main parameters
    parser.add_argument("--model_path", type=str, default="./model_training/", help="Experiment dump path")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--table_path", type=str, default="", help="table path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--title_path", type=str, default="", help="title path")

    parser.add_argument("--beam_size", type=int, default=1, help="beam size")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="")
    parser.add_argument("--early_stopping", type=bool, default=False, help="")

    return parser


def main(params):
    # generate parser / parse parameters
    #parser = get_parser()
    #params = parser.parse_args()
    reloaded = torch.load(params.model_path)

    model_params = AttrDict(reloaded['params'])

    # update dictionary parameters
    for name in ['src_n_words', 'tgt_n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))
    # print(f'src {getattr(model_params, "src_n_words")}')
    # print(f'tgt {getattr(model_params, "tgt_n_words")}')
    # build dictionary / build encoder / build decoder / reload weights
    source_dico = Dictionary(reloaded['source_dico_id2word'], reloaded['source_dico_word2id'])
    target_dico = Dictionary(reloaded['target_dico_id2word'], reloaded['target_dico_word2id'])
    # originalDecoder = reloaded['decoder'].copy()
    encoder = TransformerEncoder(model_params, source_dico, with_output=False).cuda().eval()
    encoder.load_state_dict(reloaded['encoder'])
    decoder = TransformerDecoder(model_params, target_dico, with_output=True).cuda().eval()
    decoder.load_state_dict(reloaded['decoder'])
    # read sentences from stdin
    table_lines = []
    title_lines = []
    table_inf = open(params.table_path, 'r', encoding='utf-8')
    for table_line in table_inf:
        table_lines.append(table_line)
    with open(params.title_path, 'r', encoding='utf-8') as title_inf:
        for title_line in title_inf:
            title_lines.append(title_line)

    assert len(title_lines) == len(table_lines)

    outf = io.open(params.output_path, 'w', encoding='utf-8')

    fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
               'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', 'with']

    for i in range(0, len(table_lines), params.batch_size):
        # prepare batch

        """valueLengths = []
        xLabelLengths = []
        yLabelLengths = []
        titleLengths = []"""
        enc_x1_ids = []
        enc_x2_ids = []
        enc_x3_ids = []
        enc_x4_ids = []
        for table_line, title_line in zip(table_lines[i:i + params.batch_size], title_lines[i:i + params.batch_size]):
            record_seq = [each.split('|') for each in table_line.split()]
            assert all([len(x) == 4 for x in record_seq])

            enc_x1_ids.append(torch.LongTensor([source_dico.index(x[0]) for x in record_seq]))
            enc_x2_ids.append(torch.LongTensor([source_dico.index(x[1]) for x in record_seq]))
            enc_x3_ids.append(torch.LongTensor([source_dico.index(x[2]) for x in record_seq]))
            enc_x4_ids.append(torch.LongTensor([source_dico.index(x[3]) for x in record_seq]))

            xLabel = record_seq[1][0].split('_')
            yLabel = record_seq[0][0].split('_')
            """cleanXLabel = len([item for item in xLabel if item not in fillers])
            cleanYLabel = len([item for item in yLabel if item not in fillers])
            cleanTitle = len([word for word in title_line.split() if word not in fillers])

            xLabelLengths.append(cleanXLabel)
            yLabelLengths.append(cleanYLabel)
            titleLengths.append(cleanTitle)
            valueLengths.append(round(len(record_seq)/2))"""

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
            # print(sent)
            # output translation
            # source = table_lines[i + j].strip()
            # print(source)
            tokens = []
            for k in range(len(sent)):
                ids = sent[k].item()
                #if ids in removedDict:
                #    print('index error')
                word = target_dico[ids]
                tokens.append(word)
            target = " ".join(tokens)
            sys.stderr.write("%i / %i: %s\n" % (i + j, len(table_lines), target))
            outf.write(target + "\n")
    outf.close()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    #params = argparse.Namespace(batch_size=1, beam_size=4, early_stopping=False, length_penalty=1.0, model_path='may21gelu-80.pth',
    #                            output_path='results/may26/templateOutput_529gp80_beam=4_batch=1.txt',
    #                            table_path='data/test/testData.txt', title_path='data/test/testTitle.txt')
    # check parameters
    print(params)
    assert os.path.isfile(params.model_path)
#    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)

    """ # max_len = int(1.5 * enc_xlen.max().item() + 10)
         maxLengthXLabel = max(xLabelLengths)
         maxLengthYLabel = max(yLabelLengths)
         maxLengthValue = max(valueLengths)
         maxLengthTitle = max(titleLengths)

         removedDict = {}
         for n in range(maxLengthXLabel, 10):
             deleteThis = f'templateXLabel[{n}]'
             if target_dico.__contains__(deleteThis):
                 ids = target_dico.index(deleteThis)
                 removedDict[ids] = deleteThis
         for n in range(maxLengthYLabel, 10):
             deleteThis = f'templateYLabel[{n}]'
             if target_dico.__contains__(deleteThis):
                 ids = target_dico.index(deleteThis)
                 removedDict[ids] = deleteThis
         for n in range(maxLengthTitle, 15):
             deleteThis = f'templateTitle[{n}]'
             if target_dico.__contains__(deleteThis):
                 ids = target_dico.index(deleteThis)
                 removedDict[ids] = deleteThis
         for n in range(maxLengthValue, 31):
             deleteThis = f'templateXValue[{n}]'
             if target_dico.__contains__(deleteThis):
                 ids = target_dico.index(deleteThis)
                 removedDict[ids] = deleteThis
             deleteThis = f'templateYValue[{n}]'
             if target_dico.__contains__(deleteThis):
                 ids = target_dico.index(deleteThis)
                 removedDict[ids] = deleteThis

         print(removedDict)

         #update prediction layer weights for removed tokens in the decoder
         #test changing pred layer bias to -100 for each token out of vocab
         newDecoder = originalDecoder.copy()
         for key in newDecoder.keys():
             if key == 'pred_layer.proj.bias':
                 for ids in removedDict.keys():
                     newDecoder["pred_layer.proj.bias"][ids] = torch.tensor(-100).cuda()

         #update decoder weights
         decoder = TransformerDecoder(model_params, target_dico, with_output=True).cuda().eval()
         decoder.load_state_dict(newDecoder.copy())"""