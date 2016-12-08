# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import random
import string

import numpy as np
from six.moves import xrange
import tensorflow as tf

import kenlm

import nlc_model
import nlc_data
from util import get_tokenizer

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_string("lmfile", None, "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0.3, "Language model relative weight.")

FLAGS = tf.app.flags.FLAGS
reverse_vocab, vocab = None, None
lm = None

def create_model(session, vocab_size, forward_only):
  model = nlc_model.NLCModel(
      vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def tokenize(sent, vocab, depth=FLAGS.num_layers):
  align = pow(2, depth - 1)
  token_ids = nlc_data.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS))
  ones = [1] * len(token_ids)
  pad = (align - len(token_ids)) % align

  token_ids += [nlc_data.PAD_ID] * pad
  ones += [0] * pad

  source = np.array(token_ids).reshape([-1, 1])
  mask = np.array(ones).reshape([-1, 1])

  return source, mask


def detokenize(sents, reverse_vocab):
  # TODO: char vs word
  def detok_sent(sent):
    outsent = ''
    for t in sent:
      if t >= len(nlc_data._START_VOCAB):
        outsent += reverse_vocab[t]
    return outsent
  return [detok_sent(s) for s in sents]


def lm_rank(strs, probs):
  if lm is None:
    return strs[0]
  a = FLAGS.alpha
  lmscores = [lm.score(s)/(1+len(s.split())) for s in strs]
  probs = [ p / (len(s)+1) for (s, p) in zip(strs, probs) ]
  for (s, p, l) in zip(strs, probs, lmscores):
    print(s, p, l)

  rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
  rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
  generated = strs[rerank[-1]]
  lm_score = lmscores[rerank[-1]]
  nw_score = probs[rerank[-1]]
  score = rescores[rerank[-1]]
  return generated #, score, nw_score, lm_score

#  if lm is None:
#    return strs[0]
#  a = FLAGS.alpha
#  rescores = [(1-a)*p + a*lm.score(s) for (s, p) in zip(strs, probs)]
#  rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x:x[1])]
#  return strs[rerank[-1]]

def decode_beam(model, sess, encoder_output, max_beam_size):
  toks, probs = model.decode_beam(sess, encoder_output, beam_size=max_beam_size)
  return toks.tolist(), probs.tolist()

def fix_sent(model, sess, sent):
  # Tokenize
  input_toks, mask = tokenize(sent, vocab)
  # Encode
  encoder_output = model.encode(sess, input_toks, mask)
  # Decode
  beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size)
  # De-tokenize
  beam_strs = detokenize(beam_toks, reverse_vocab)
  # Language Model ranking
  best_str = lm_rank(beam_strs, probs)
  # Return
  return best_str

def decode():
  # Prepare NLC data.
  global reverse_vocab, vocab, lm

  if FLAGS.lmfile is not None:
    print("Loading Language model from %s" % FLAGS.lmfile)
    lm = kenlm.LanguageModel(FLAGS.lmfile)

  print("Preparing NLC data in %s" % FLAGS.data_dir)

  x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
    FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
    tokenizer=get_tokenizer(FLAGS))
  vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path)
  vocab_size = len(vocab)
  print("Vocabulary size: %d" % vocab_size)

  with tf.Session() as sess:
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)

    while True:
      sent = raw_input("Enter a sentence: ")

      output_sent = fix_sent(model, sess, sent)

      print("Candidate: ", output_sent)

def main(_):
  decode()

if __name__ == "__main__":
  tf.app.run()
