Introduction
============

This porject is porting our original code in Theano to Tensorflow. Still in very early stage and under heavy development.


INTRODUCTION
============

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow

DEPENDENCIES
============

Tensorflow 0.9


TRAINING
========

To train character level model (default):

   $ python train.py


To train word level model:

   $ python train.py --tokenizer WORD


INTERACTIVE DECODING
====================

   $ python decode.py
