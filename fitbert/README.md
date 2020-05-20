# FitBERT

![buff bert](img/fitbert.png)

FitBert ((F)ill (i)n (t)he blanks, (BERT)) is a library for using [BERT](https://arxiv.org/abs/1810.04805) to fill in the blank(s) in a section of text from a list of options. Here is the envisioned usecase for FitBert:

1. A service (statistical model or something simpler) suggests replacements/corrections for a segment of text
2. That service is specialized to a domain, and isn't good at the big picture, e.g. grammar
3. That service passes the segment of text, with the words to be replaced identified, and the list of suggestions
4. FitBert _crushes_ all but the best suggestion :muscle:

[Blog post walkthrough](https://medium.com/@samhavens/introducing-fitbert-4b047af860fd)

## Installation

## License

This software is distributed under the Apache 2.0 license, except for the WordNet lemma data used for delemmatization, which is distributed with its original license, which is located in `./fitbert/data/LICENSE`.

## From PyPi

`pip install fitbert`

## Usage

[A Jupyter notebook with a short introduction is available here.](https://colab.research.google.com/drive/1WrYzy9l_arpnTlhCCKViiilPe4WKZJjq)

FitBert will automatically use GPU if `torch.cuda.is_available()`. Or when you instantiate it, you can pass `FitBert(model_name="distilbert-base-uncased", disable_gpu=True)`. Fastest batches are using distilbert on CPU with batch size one, maximum throughput is with GPU and larger batches.

### Usage as a library / in a server

```python
from fitbert import FitBert


# currently supported models: bert-large-uncased and distilbert-base-uncased
# this takes a while and loads a whole big BERT into memory
fb = FitBert()

masked_string = "Why Bert, you're looking ***mask*** today!"
options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options)
# >>> ['handsome', 'strong', 'buff']
# or
filled_in = fb.fitb(masked_string, options=options)
# >>> "Why Bert, you're looking handsome today!"
```

We commonly find ourselves knowing what verb to suggest, but not what conjugation:

```python
from fitbert import FitBert


fb = FitBert()

masked_string = "Why Bert, you're ***mask*** handsome today!"
options = ['looks']

filled_in = fb.fitb(masked_string, options=options)
# >>> "Why Bert, you're looking handsome today!"

# under the hood, we notice there is only one suggestion and act as if
# fitb was called with delemmatize=True:
filled_in = fb.fitb(masked_string, options=options, delemmatize=True)
```

If you are already using `pytorch_pretrained_bert.BertForMaskedLM`, or `transformers.BertForMaskedLM` and have an instance of BertForMaskedLM already instantiated, you can pass pass it in to reuse it:

```python
BLM = pytorch_pretrained_bert.BertForMaskedLM.from_pretrained(model_name)
# or
BLM = transfomers.BertForMaskedLM.from_pretrained(model_name)
fb = FitBert(model=BLM)
```

You can also have FitBert mask the string for you

```python
from fitbert import FitBert


fb = FitBert()

unmasked_string = "Why Bert, you're looks handsome today!"
span_to_mask = (17, 22)
masked_string, masked = fb.mask(unmasked_string, span_to_mask)
# >>> "Why Bert, you're ***mask*** handsome today!", 'looks'

# you can set options = [masked] or use any List[str]
options = [masked]

filled_in = fb.fitb(masked_string, options=options)
# >>> "Why Bert, you're looking handsome today!"
```

and there is a convenience method for doing this:

```python
unmasked_string = "Why Bert, you're looks handsome today!"
span_to_mask = (17, 22)

filled_in = fb.mask_fitb(unmasked_string, span_to_mask)
# >>> "Why Bert, you're looking handsome today!"
```

### Client

If you are sending strings to a FitBert server, you need to either mask the string yourself, or identify the span you want masked:

```python
from fitbert import FitBert

s = "This might be justified as a means of signalling the connection between drunken driving and fatal accidents."

better_string, span_to_change = MyRuleBasedNLPModel.remove_overly_fancy_language(s)

assert better_string == "This might be justified to signalling the connection between drunken driving and fatal accidents.", "Notice 'as a means of' became 'to', but we didn't re-conjuagte signalling, or fix the spelling mistake"

assert span_to_change == (27, 37), "This span is the start and stop of the characters for the substring 'signalling'."

masked_string, replaced_substring = FitBert.mask(better_string, span_to_change)

assert masked_string == "This might be justified to ***mask*** the connection between drunken driving and fatal accidents."

assert replaced_substring == "signalling"

FitBertServer.fitb(masked_string, options=[replaced_substring])
```

The benefit to doing this over masking yourself is that if the internally used masking token changes, you don't have to know about that. Also, you don't need to make an instance of FitBert, so you don't have to incur the cost of downloading a pretrained Bert model.

However, you could also write your `CallFitBertServer` function to take an unmasked string and a span, something like:

```python
FitBertServer.mask_fitb(better_string, span_to_change)
```

And then not need to install `FitBert` in your client at all.

## Development

Run tests with `python -m pytest` or `python -m pytest -m "not slow"` to skip the 20 seconds of loading pretrained bert.

### Acknowledgement

Thanks to [NodoBird](https://instagram.com/nodobird?igshid=lqt5h1uicxsy) for letting us use the awesome portrait of Bert depicted above.
