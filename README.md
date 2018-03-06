# PTB Language Modelling task with RNNS(LSTM,GRU)
This repository is used for a language modelling pareto competition at TTIC. 

TODO:

Upload files


### Competition

(time ratio, perplexity)

time ratio = training time/ base model training time. (The base model is trained with default params and trained on a single CPU with rougly 1 hour)

Note: Your training time must be calculated from a single CPU.

## Model
I implemented Sampled Softmax method to the originial RNNS model. In addition, an implementation of using pre-trained word embedding with size of 200 and 300 from GloVe can be found in the main.py. The model is also trained with Adagrad optimizer.

## Software Requirements

This codebase requires Python 3, [PyTorch](http://pytorch.org/)

## GloVe

Please download the GloVe from here: [word2vec-api](https://github.com/3Top/word2vec-api)

or

download from here directly: [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip).

## Usage

```bash
python generate.py                      # Generate samples from the trained LSTM model.
```

## Acknowledge
This repository contains the code originally forked from the [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) that is modified to present attention layer into the model.
