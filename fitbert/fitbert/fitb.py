from collections import defaultdict
from typing import Dict, List, Tuple, Union, overload

import torch
from fitbert.delemmatize import Delemmatizer
from fitbert.utils import mask as _mask
from functional import pseq, seq
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)


class FitBert:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name="bert-large-uncased",
        mask_token="***mask***",
        disable_gpu=False,
    ):
        self.mask_token = mask_token
        self.delemmatizer = Delemmatizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        print("using model:", model_name)
        print("device:", self.device)

        if not model:
            if "distilbert" in model_name:
                self.bert = DistilBertForMaskedLM.from_pretrained(model_name)
            else:
                self.bert = BertForMaskedLM.from_pretrained(model_name)
            self.bert.to(self.device)
        else:
            self.bert = model
            self.bert.to(self.device)

        if not tokenizer:
            if "distilbert" in model_name:
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        self.bert.eval()

    @staticmethod
    def softmax(x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    @staticmethod
    def is_multi(options: List[str]) -> bool:
        return seq(options).filter(lambda x: len(x.split()) != 1).non_empty()

    def mask(self, s: str, span: Tuple[int, int]) -> Tuple[str, str]:
        return _mask(s, span, mask_token=self.mask_token)

    def _tokens_to_masked_ids(self, tokens, mask_ind):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = "[MASK]"
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        return masked_ids

    def _get_sentence_probability(self, sent: str) -> float:

        tokens = self.tokenizer.tokenize(sent)
        input_ids = (
            seq(tokens)
            .enumerate()
            .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i))
            .list()
        )

        tens = torch.tensor(input_ids).to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            prob = (
                seq(tokens_ids)
                .enumerate()
                .starmap(lambda i, x: float(probs[i][i + 1][x].item()))
                .reduce(lambda x, y: x * y, 1)
            )

            del tens, preds, probs, tokens, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return prob

    def _delemmatize_options(self, options: List[str]) -> List[str]:
        options = (
            seq(options[:])
            .flat_map(lambda x: self.delemmatizer(x))
            .union(options)
            .list()
        )
        return options

    def guess_single(self, masked_sent: str, n: int = 1):

        pre, post = masked_sent.split(self.mask_token)

        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)

            pred_top = torch.topk(probs[0, target_idx], n)
            pred_prob = pred_top[0].tolist()
            pred_idx = pred_top[1].tolist()

            pred_tok = self.tokenizer.convert_ids_to_tokens(pred_idx)

            del pred_top, pred_idx, tens, preds, probs, input_ids, tokens
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return pred_tok, pred_prob

    def rank_single(self, masked_sent: str, words: List[str]):

        pre, post = masked_sent.split(self.mask_token)

        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]

        words_ids = (
            seq(words)
            .map(lambda x: self.tokenizer.tokenize(x))
            .map(lambda x: self.tokenizer.convert_tokens_to_ids(x)[0])
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)

            ranked_pairs = (
                seq(words_ids)
                .map(lambda x: float(probs[0][target_idx][x].item()))
                .zip(words)
                .sorted(key=lambda x: x[0], reverse=True)
            )

            ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
            ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

            del tens, preds, probs, tokens, words_ids, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return ranked_options, ranked_options_prob

    def rank_multi(self, masked_sent: str, options: List[str]):
        ranked_pairs = (
            seq(options)
            .map(lambda x: masked_sent.replace(self.mask_token, x))
            .map(lambda x: self._get_sentence_probability(x))
            .zip(options)
            .sorted(key=lambda x: x[0], reverse=True)
        )
        ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
        ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()
        return ranked_options, ranked_options_prob

    def _simplify_options(self, sent: str, options: List[str]):

        options_split = seq(options).map(lambda x: x.split())

        trans_start = list(zip(*options_split))

        start = (
            seq(trans_start)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        options_split_reversed = seq(options_split).map(
            lambda x: seq(x[len(start) :]).reverse()
        )

        trans_end = list(zip(*options_split_reversed))

        end = (
            seq(trans_end)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        start_words = seq(start).make_string(" ")
        end_words = seq(end).reverse().make_string(" ")

        options = (
            seq(options_split)
            .map(lambda x: x[len(start) : len(x) - len(end)])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )

        sub = seq([start_words, self.mask_token, end_words]).make_string(" ").strip()
        sent = sent.replace(self.mask_token, sub)

        return options, sent, start_words, end_words

    def rank(
        self,
        sent: str,
        options: List[str],
        delemmatize: bool = False,
        with_prob: bool = False,
    ):
        """
        Rank a list of candidates

        returns: Either a List of strings,
        or if `with_prob` is True, a Tuple of List[str], List[float]

        """

        options = seq(options).distinct().list()

        if delemmatize:
            options = seq(self._delemmatize_options(options)).distinct().list()

        if seq(options).len() == 1:
            return options

        options, sent, start_words, end_words = self._simplify_options(sent, options)

        if self.is_multi(options):
            ranked, prob = self.rank_multi(sent, options)
        else:
            ranked, prob = self.rank_single(sent, options)

        ranked = (
            seq(ranked)
            .map(lambda x: [start_words, x, end_words])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )
        if with_prob:
            return ranked, prob
        else:
            return ranked

    def rank_with_prob(self, sent: str, options: List[str], delemmatize: bool = False):
        ranked, prob = self.rank(sent, options, delemmatize, True)
        return ranked, prob

    def guess(self, sent: str, n: int = 1) -> List[str]:
        pred_tok, _ = self.guess_single(sent, n)
        return pred_tok

    def guess_with_prob(self, sent: str, n: int = 1):
        pred_tok, pred_prob = self.guess_single(sent, n)
        return pred_tok, pred_prob

    def fitb(self, sent: str, options: List[str], delemmatize: bool = False) -> str:
        ranked = self.rank(sent, options, delemmatize)
        best_word = ranked[0]
        return sent.replace(self.mask_token, best_word)

    def mask_fitb(self, sent: str, span: Tuple[int, int]) -> str:
        masked_str, replaced = self.mask(sent, span)
        options = [replaced]
        return self.fitb(masked_str, options, delemmatize=True)
