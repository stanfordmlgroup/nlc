This project ports our original code in Theano to Tensorflow. Still under development.

# Introduction

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow.

# Training

To train character level model (default):

   $ python train.py

To train word level model:

   $ python train.py --tokenizer WORD

# Interactive decoding

   $ python decode.py

# Other implementations

- Chainer implementation by @sotetsuk: [https://github.com/sotetsuk/neural-language-correction](https://github.com/sotetsuk/neural-language-correction)
