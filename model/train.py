# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import argparse

from src.data.loader import load_data
from src.utils import bool_flag, initialize_exp
from src.model import check_model_params, build_model
from src.trainer import EncDecTrainer, SingleTrainer
from src.evaluation.evaluator import EncDecEvaluator, SingleEvaluator

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Data2text generation")

    # main parameters
    parser.add_argument("--model_path", type=str, default="./experiments/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--save_periodic", type=int, default=1,
                        help="Save the model periodically (0 to disable)")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=False,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--enc_n_layers", type=int, default=2,
                        help="Number of Transformer layers")
    parser.add_argument("--dec_n_layers", type=int, default=6,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--share_srctgt_emb", type=bool_flag, default=False,
                        help="Share source and target embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--encoder_positional_emb", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")

    # adaptive softmax
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="label smoothing parameter")
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    parser.add_argument("--cs_step", type=bool_flag, default=False,
                        help="content selection step")
    parser.add_argument("--lm_step", type=bool_flag, default=False,
                        help="language modeling step")
    parser.add_argument("--sm_step", type=bool_flag, default=False,
                        help="summarization step")
    parser.add_argument("--sm_step_with_cc_loss", type=bool_flag, default=False,
                        help="summarization step with conditional copy loss")
    parser.add_argument("--sm_step_with_cs_proba", type=bool_flag, default=False,
                        help="summarization step with conditional copy loss")
    # dataOld
    parser.add_argument("--train_cs_table_path", type=str, default="",
                        help="train content selection table dataOld pth")
    parser.add_argument("--train_sm_table_path", type=str, default="",
                        help="train summarization table dataOld pth")
    parser.add_argument("--train_sm_summary_path", type=str, default="",
                        help="train summarization summary dataOld pth")
    parser.add_argument("--valid_table_path", type=str, default="",
                        help="valid table dataOld pth")
    parser.add_argument("--valid_summary_path", type=str, default="",
                        help="valid summary dataOld pth")

    # batch parameters
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split dataOld across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=10000,
                        help="Epoch size / evaluation frequency (-1 for parallel dataOld size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")

    # training coefficients
    parser.add_argument("--lambda_cs", type=str, default="0",
                        help="content selection coefficient")
    parser.add_argument("--lambda_sm", type=str, default="0",
                        help="summarization coefficient")
    parser.add_argument("--lambda_lm", type=str, default="0",
                        help="language modeling coefficient")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_model_strict", type=bool_flag, default=False,
                        help="reload model strict")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--beam_size", type=int, default=1,
                        help="beam size in beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="length penalty in beam search")
    parser.add_argument("--early_stopping", type=bool_flag, default=True,
                        help="early stopping in beam search")
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="early stopping in beam search")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_cs", type=bool_flag, default=False,
                        help="eval cs")
    return parser

def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # load dataOld
    data = load_data(params)
    # check_vocab(dataOld)

    # build model
    if params.encoder_only:
        model = build_model(params, data['source_dico'])
    else:
        encoder, decoder = build_model(params, data['source_dico'], data['target_dico'])


    # build trainer, reload potential checkpoints / build evaluator
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_iter = 0

        while trainer.n_iter < trainer.epoch_size:
            if params.cs_step:
                trainer.content_selection_step(params.lambda_cs)
            if params.sm_step:
                trainer.summarization_step(params.lambda_sm)
            if params.lm_step:
                trainer.clm_step(params.lambda_lm)
            trainer.iter()
        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer)
        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    # check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
