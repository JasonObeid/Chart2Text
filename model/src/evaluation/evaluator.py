# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from . import summaryComparison_bleuReverse as summaryComparison
from ..utils import to_cuda, restore_segmentation, concat_batches


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)

test_list = ['valid']

logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.params = params

        if params.eval_bleu:
            params.hyp_path = os.path.join(params.model_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()

    def get_iterator(self, task, data_set):
        #print(data_set)
        #print(self.data)
        """
        Create a new iterator for a dataset.
        """
        # assert data_set in test_list

        n_sentences = -1
        subsample = 1

        iterator = self.data[task][data_set].get_iterator(
            shuffle=False,
            group_by_size=False,
            n_sentences=n_sentences
        )

        for batch in iterator:
            #print(batch)
            yield batch #if lang2 is None or lang1 < lang2 else batch[::-1]

    """
    def mask_out(self, x, lengths, rng):
        #Decide of random words to mask out.
        #We specify the random generator to ensure that the test is the same at each epoch.
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask
    """

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():
            for data_set in test_list:
                if params.encoder_only:
                    #self.evaluate_clm(scores, data_set)
                    #self.evaluate_clm(scores, 'train')
                    self.evaluate_cs(scores, data_set)
                    self.evaluate_cs(scores, 'train')
                else:
                    self.evaluate_mt(scores, data_set, params.eval_bleu)
        return scores

    """def evaluate_clm(self, scores, data_set):
        
        #Evaluate perplexity and next word prediction accuracy.

        params = self.params
        assert data_set in test_list or data_set == 'train'

        self.encoder.eval()
        encoder = self.encoder
        params = params

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator('lm', data_set):
            # generate batch
            (table_entities, table_types, table_values, table_feats, table_labels) = batch
            x1, lengths = table_entities
            x2, _ = table_types
            x3, _ = table_values
            x4, _ = table_feats
            y, _ = table_labels

            bs = lengths.size(0)
            if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
                self.n_sentences += params.batch_size
                return

            # cuda
            if params.cuda:
                x1, x2, x3, x4, lengths, y = to_cuda(x1, x2, x3, x4, lengths, y)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = encoder('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = encoder('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # log
        logger.info("Found %i words in %s. %i were predicted correctly." % (n_words, data_set, n_valid))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_clm_ppl' % (data_set)
        acc_name = '%s_clm_acc' % (data_set)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words"""

    def evaluate_cs(self, scores, data_set):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in test_list or data_set == 'train'

        self.encoder.eval()
        encoder = self.encoder
        params = params

        n_words = 0
        n_preds = 0
        n_valid = 0

        for batch in self.get_iterator('cs', data_set):
            # generate batch
            (table_entities, table_types, table_values, table_feats, table_labels) = batch
            x1, lengths = table_entities
            x2, _ = table_types
            x3, _ = table_values
            x4, _ = table_feats
            y, _ = table_labels

            bs = lengths.size(0)
            if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
                self.n_sentences += params.batch_size
                return

            # cuda
            if params.cuda:
                x1, x2, x3, x4, lengths, y = to_cuda(x1, x2, x3, x4, lengths, y)

            # encode source sentence
            encoder_output = encoder('fwd', x1=x1, x2=x2, x3=x3, x4=x4, lengths=lengths)
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1  # do not predict anything given the last <eos> token 
            pred_mask[0,:] = 0 # do not predict anything given the first <eos> token
            y = y[pred_mask]
            enc_scores, loss = encoder('predict', tensor=encoder_output, pred_mask=pred_mask, y=y)
            output = (enc_scores > 0.5).squeeze().float()
            pos_cnt = output.sum().item()
            pos_ref = y.sum().item()
            pos_correct = sum([(x==y==1) for x, y in zip(output.tolist(), y.tolist())])

            n_valid += pos_correct
            n_words += pos_ref
            n_preds += pos_cnt

        # compute perplexity and prediction accuracy
        scores['%s_cs_prec' % (data_set)] = 100. * n_valid / n_preds
        scores['%s_cs_recall' % (data_set)] = 100. * n_valid / n_words
        scores['%s_cs_f1' % (data_set)] = 2 * scores['%s_cs_prec' % (data_set)] * scores['%s_cs_recall' % (data_set)] / (scores['%s_cs_prec' % (data_set)] + scores['%s_cs_recall' % (data_set)])


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.source_dico = data['source_dico']
    
class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.source_dico = data['source_dico']
        self.target_dico = data['target_dico']
        super().__init__(trainer, data, params)

    def evaluate_mt(self, scores, data_set, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in test_list

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder
        decoder = self.decoder

        params = params

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator('sm', data_set):
            # generate batch
            (table_entities, table_types, table_values, table_feats, 
                table_labels, summaries, summary_labels) = batch
            x11, len1 = table_entities
            x12, _ = table_types
            x13, _ = table_values
            x14, _ = table_feats
            y11, _ = table_labels

            x2, len2 = summaries
            copy_label, _ = summary_labels

            vocab_mask = (torch.sum(torch.nn.functional.one_hot(x11, params.tgt_n_words), dim=0) > 0)
            vocab_mask = vocab_mask.repeat(x2.size()[0],1,1)
            assert(vocab_mask.size()[:2] == x2.size()[:2])

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y2 = x2[1:].masked_select(pred_mask[:-1])
            copy_label = copy_label[1:].masked_select(pred_mask[:-1])
            copy_label = copy_label.byte()
            assert len(y2) == (len2 - 1).sum().item()
            vocab_mask = vocab_mask[1:].masked_select(pred_mask[:-1].unsqueeze(-1)).view(-1, params.tgt_n_words)
            vocab_mask = vocab_mask.byte()

            # cuda
            if params.cuda:
                x11, x12, x13, x14, len1, x2, len2, y2, copy_label, vocab_mask = \
                    to_cuda(x11, x12, x13, x14, len1, x2, len2, y2, copy_label, vocab_mask)

            # encode source sentence
            encoder_output = encoder('fwd', x1=x11, x2=x12, x3=x13, x4=x14, lengths=len1)
            if params.sm_step_with_cs_proba:
                cs_scores = self.encoder('score', tensor=encoder_output) 
                encoder_output = encoder_output * cs_scores

            encoder_output = encoder_output.transpose(0, 1)
            # decode target sentence
            decoder_output = decoder('fwd', x=x2, lengths=len2, causal=True, 
                                      src_enc=encoder_output, src_len=len1)
            # loss
            if params.sm_step_with_cc_loss:
                word_scores, word_loss = decoder('predict', tensor=decoder_output, pred_mask=pred_mask, 
                                        y=y2, keep_dim=True)

                copy_scores = word_scores * vocab_mask.float() + (1 - vocab_mask.float()) * torch.min(word_scores)

                copy_probas, copy_loss = decoder('predict_copy', tensor=decoder_output, pred_mask=pred_mask, 
                                                    y=copy_label)
                copy_scores = copy_scores * copy_probas
                word_scores = word_scores * (1.0 - copy_probas)
                copy_label = copy_label.unsqueeze(-1).expand_as(copy_scores)
                word_scores = torch.where(copy_label, copy_scores, word_scores)
                loss = F.cross_entropy(word_scores, y2, reduction='mean')
            else:
                word_scores, loss = decoder('predict', tensor=decoder_output, pred_mask=pred_mask, y=y2)

            # update stats
            n_words += y2.size(0)
            xe_loss += loss.item() * len(y2)
            n_valid += (word_scores.max(1)[1] == y2).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                #vocab_mask = (torch.sum(torch.nn.functional.one_hot(x11, params.tgt_n_words), dim=0) > 0)
                #vocab_mask = vocab_mask.byte()

                # max_len = int(1.5 * len1.max().item() + 10)
                max_len = 602
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(encoder_output, len1, max_len=max_len, vocab_mask=None)
                else:
                    generated, lengths = decoder.generate_beam(
                        encoder_output, len1, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.target_dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_mt_ppl' % (data_set)] = np.exp(xe_loss / n_words)
        scores['%s_mt_acc' % (data_set)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            #Templated
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}.txt'.format(scores['epoch'], data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = 'data/valid/validOriginalSummary.txt'
            print(ref_path)
            temp_path = os.path.join(params.hyp_path, 'temp_hyp.txt')
            # export sentences to temp hypothesis file / restore BPE segmentation
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(temp_path)
            # reverse templating and save in persistent hypothesis file
            dataPath = 'data/valid/validData.txt'
            titlePath = 'data/valid/validTitle.txt'
            try:
                summaryComparison.run(temp_path, hyp_path, dataPath, titlePath)
            except:
                logger.log('summary reversal failed')
                hyp_path = temp_path
            #untemplated
            """# hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}.txt'.format(scores['epoch'], data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = 'data_templated/valid/validOriginalSummary.txt'

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)"""

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_mt_bleu' % (data_set)] = bleu

def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    #print(f'decoded: {sentences}')
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
