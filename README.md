# PTB Language Modelling task with RNNS(LSTM,GRU) + Sampled Softmax + Pre-trained Word Embedding(GloVe)

This repository is used for a language modelling pareto competition at TTIC. 
I implemented Sampled Softmax method to the originial RNNS model. In addition, an implementation of using pre-trained word embedding with size of 200 and 300 from GloVe can be found in the main.py.

## Software Requirements

This codebase requires Python 3, [PyTorch](http://pytorch.org/)

## GloVe

Please download the GloVe from here. [word2vec-api](https://github.com/3Top/word2vec-api)

or
download from here directly [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip).

## Usage

```bash
python main.py --soft --adagrad --lr 0.01        # Train a LSTM on PTB with sampled softmax and using Adagrad as the optimizer with learning rate = 0.01
python main.py --pre --emsize 300       # Train a LSTM on PTB with pre-trained embedding with emsize 300
python generate.py                      # Generate samples from the trained LSTM model.
```

## Acknowledge
This repository contains the code originally forked from the [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) that is modified to present attention layer into the model.
