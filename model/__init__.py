# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerDecoder, TransformerEncoder


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, src_dico, tgt_dico=None):
    """
    Build model.
    """
    if params.encoder_only:
        # build
        encoder = TransformerEncoder(params, src_dico, with_output=True)
        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            if params.cuda:
                data = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(0))
            else:
                data = torch.load(params.reload_model, map_location=lambda storage, loc: 'cpu')
            encoder.load_state_dict(data['encoder'])
        logger.debug("Encoder: {}".format(encoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        if params.cuda:
            return encoder.cuda()
        else:
            return encoder
    else:
        encoder = TransformerEncoder(params, src_dico, with_output=True)
        decoder = TransformerDecoder(params, tgt_dico, with_output=True)

        if params.share_srctgt_emb:
            assert encoder.n_words == decoder.n_words
            encoder.embeddings.weight = decoder.embeddings.weight

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            if params.cuda:
                data = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(0))
            else:
                data = torch.load(params.reload_model, map_location=lambda storage, loc: 'cpu')

            encoder.load_state_dict(data['encoder'])
            decoder.load_state_dict(data['decoder'], strict=params.reload_model_strict)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        if params.cuda:
            return encoder.cuda(), decoder.cuda()
        else:
            return encoder, decoder
