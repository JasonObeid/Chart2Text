 # Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import get_optimizer, to_cuda
from .utils import parse_lambda_config, update_lambdas

logger = getLogger()

class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        self.stopping_criterion = None
        self.best_stopping_criterion = None

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('cs', []), ('lambda_cs', 0)] +
            [('sm', []), ('lambda_sm', 0)] +
            [('lm', []), ('lambda_lm', 0)]
        )

        # dataOld iterators
        self.iterators = {}

        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

    def get_optimizer_fp(self, module):
        """
        Build optimizer.
        """
        assert module in ['model', 'encoder', 'decoder']
        optimizer = get_optimizer(getattr(self, module).parameters(), self.params.optimizer)
        return optimizer

    def optimize(self, loss, modules):
        """
        Optimize.
        """
        if type(modules) is str:
            modules = [modules]

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # zero grad
        for module in modules:
            self.optimizers[module].zero_grad()

        loss.backward()

        # clip gradients
        if self.params.clip_grad_norm > 0:
            for module in modules:
                clip_grad_norm_(getattr(self, module).parameters(), self.params.clip_grad_norm)

        # optimization step
        for module in modules:
            self.optimizers[module].step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_iter % 10 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f} (coef={:.4f})'.format(k, np.mean(v), self.stats['lambda_'+k]) 
                                                for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizers[self.MODEL_NAMES[0]].param_groups[0]['lr']
        s_lr = " - Transformer LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training dataOld iterator (%s) ..." % iter_name)
        iterator = self.data[iter_name]['train'].get_iterator(
            shuffle=True,
            group_by_size=self.params.group_by_size,
            n_sentences=-1,
        )
        self.iterators[iter_name] = iterator
        return iterator

    def get_batch(self, iter_name):
        """
        Return a batch of sentences from a dataset.
        """
        #assert stream is False or lang2 is None
        iterator = self.iterators.get(iter_name, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name)
            x = next(iterator)
        return x
        # return x if lang2 is None or lang1 < lang2 else x[::-1]

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.model_path, '%s.pth' % name)
        logger.info('Saving models to %s ...' % path)
        data = {}
        for name in self.MODEL_NAMES:
            data[name] = getattr(self, name).state_dict()

        if self.params.encoder_only:
            data['source_dico_id2word'] = self.data['source_dico'].id2word
            data['source_dico_word2id'] = self.data['source_dico'].word2id
            data['params'] = {k: v for k, v in self.params.__dict__.items()}
        else:
            data['source_dico_id2word'] = self.data['source_dico'].id2word
            data['source_dico_word2id'] = self.data['source_dico'].word2id
            data['target_dico_id2word'] = self.data['target_dico'].id2word
            data['target_dico_word2id'] = self.data['target_dico'].word2id
            data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
        }

        for name in self.MODEL_NAMES:
            data[name] = getattr(self, name).state_dict()
            data[name + '_optimizer'] = self.optimizers[name].state_dict()

        if self.params.encoder_only:
            data['source_dico_id2word'] = self.data['source_dico'].id2word
            data['source_dico_word2id'] = self.data['source_dico'].word2id
            data['params'] = {k: v for k, v in self.params.__dict__.items()}
        else:
            data['source_dico_id2word'] = self.data['source_dico'].id2word
            data['source_dico_word2id'] = self.data['source_dico'].word2id
            data['target_dico_id2word'] = self.data['target_dico'].id2word
            data['target_dico_word2id'] = self.data['target_dico'].word2id
            data['params'] = {k: v for k, v in self.params.__dict__.items()}

        checkpoint_path = os.path.join(self.params.model_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.model_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        if self.params.cuda:
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(0))
        else:
            data = torch.load(checkpoint_path, map_location=lambda storage, loc: 'cpu')

        # reload model parameters and optimizers
        for name in self.MODEL_NAMES:
            getattr(self, name).load_state_dict(data[name])
            # getattr(self, name).load_state_dict({k[len('module.'):]: v for k, v in dataOld[name].items()})
            self.optimizers[name].load_state_dict(data[name + '_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data.get('best_metrics', 0)
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_model('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def end_epoch(self):
        """
        End the epoch.
        """
        self.save_checkpoint()
        self.epoch += 1

    def content_selection_step(self, lambda_coeff):
        """
        Parallel classification step. Predict if pairs of sentences are mutual translations of each other.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'encoder'
        model = getattr(self, name)
        model.train()

        # sample parallel sentences
        (table_entities, table_types, table_values, table_feats, table_labels) = self.get_batch('cs')
        x1, lengths = table_entities
        x2, _ = table_types
        x3, _ = table_values
        x4, _ = table_feats
        y, _ = table_labels

        bs = lengths.size(0)
        if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
            self.n_sentences += params.batch_size
            return

        if params.cuda:
            x1, x2, x3, x4, lengths, y = to_cuda(x1, x2, x3, x4, lengths, y)

        # get sentence embeddings
        encoder_output = model('fwd', x1=x1, x2=x2, x3=x3, x4=x4, lengths=lengths)
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1  # do not predict anything given the last <eos> token 
        pred_mask[0,:] = 0 # do not predict anything given the first <eos> token
        y = y[pred_mask]

        _, loss = model('predict', tensor=encoder_output, pred_mask=pred_mask, y=y)
        self.stats['cs'].append(loss.item())
        loss = lambda_coeff * loss
        # optimize
        self.optimize(loss, name)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += lengths.sum().item()
        self.stats['lambda_cs'] = lambda_coeff

    def clm_step(self, lambda_coeff):
        """
        causal language model
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'decoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        summaries, summary_label = self.get_batch('lm')
        # clm step does not use summary label
        xs, lengths = summaries
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        #if params.context_size > 0:  # do not predict without context
        #    pred_mask[:params.context_size] = 0
        y = xs[1:].masked_select(pred_mask[:-1])
        assert pred_mask.sum().item() == y.size(0)

        # cuda
        if params.cuda:
            xs, lengths, pred_mask, y = to_cuda(xs, lengths, pred_mask, y)

        # forward / loss
        tensor = model('fwd', x=xs, lengths=lengths, causal=True)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y)
        self.stats['lm'].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss, name)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()
        self.stats['lambda_lm'] = lambda_coeff

    def mlm_step(self, lang1, lang2, lambda_coeff):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'pred')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()

class SingleTrainer(Trainer):

    def __init__(self, encoder, data, params):

        self.MODEL_NAMES = ['encoder']

        # model / dataOld / params
        self.encoder = encoder
        self.data = data
        self.params = params

        # optimizers
        self.optimizers = {'encoder': self.get_optimizer_fp('encoder')}

        super().__init__(data, params)

class EncDecTrainer(Trainer):

    def __init__(self, encoder, decoder, data, params):

        self.MODEL_NAMES = ['encoder', 'decoder']

        # model / dataOld / params
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.params = params

        # optimizers
        self.optimizers = {
            'encoder': self.get_optimizer_fp('encoder'),
            'decoder': self.get_optimizer_fp('decoder'),
        }

        super().__init__(data, params)

    def summarization_step(self, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        (table_entities, table_types, table_values, table_feats, table_labels, summaries, summary_labels) = self.get_batch('sm')
        enc_x1, enc_xlen = table_entities
        enc_x2, _ = table_types
        enc_x3, _ = table_values
        enc_x4, _ = table_feats
        enc_label, _ = table_labels

        dec_x, dec_xlen = summaries

        seq_length, batch_size = dec_x.size()

        # target words to predict
        alen = torch.arange(dec_xlen.max(), dtype=torch.long, device=dec_xlen.device)
        pred_mask = alen[:, None] < dec_xlen[None] - 1  # do not predict anything given the last target word

        dec_y = dec_x[1:].masked_select(pred_mask[:-1])
        assert len(dec_y) == (dec_xlen - 1).sum().item()

        # cuda
        if params.cuda:
            enc_x1, enc_x2, enc_x3, enc_x4, enc_xlen = to_cuda(enc_x1, enc_x2, enc_x3, enc_x4, enc_xlen)
            dec_x, dec_xlen, dec_y = to_cuda(dec_x, dec_xlen, dec_y)
        
        # encode source sentence
        encoder_output = self.encoder('fwd', x1=enc_x1, x2=enc_x2, x3=enc_x3, x4=enc_x4, lengths=enc_xlen)

        if params.sm_step_with_cs_proba:
            scores = self.encoder('score', tensor=encoder_output) 
            encoder_output = encoder_output * scores

        encoder_output = encoder_output.transpose(0, 1)

        # decode target sentence
        decoder_output = self.decoder('fwd', x=dec_x, lengths=dec_xlen, causal=True, 
                                      src_enc=encoder_output, src_len=enc_xlen)

        _, loss = self.decoder('predict', tensor=decoder_output, pred_mask=pred_mask, y=dec_y)

        self.stats['sm'].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss, ['encoder', 'decoder'])

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += dec_xlen.size(0)
        self.stats['processed_w'] += (dec_xlen - 1).sum().item()
        self.stats['lambda_sm'] = lambda_coeff

