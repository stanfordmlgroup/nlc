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
tf.app.flags.DEFINE_float("alpha", 0.4, "Language model weight.")

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


def get_tokenizer(FLAGS):
  tokenizer = nlc_data.char_tokenizer if FLAGS.tokenizer.lower() == 'char' else nlc_data.basic_tokenizer
  return tokenizer


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


def detokenize(sent, reverse_vocab):
  outsent = ''
  for t in sent:
    if t >= len(nlc_data._START_VOCAB):
      outsent += reverse_vocab[t]
  return outsent


def decode_greedy(model, sess, encoder_output):
  decoder_state = None
  decoder_input = np.array([nlc_data.SOS_ID, ], dtype=np.int32).reshape([1, 1])

  attention = []
  output_sent = []
  while True:
    decoder_output, attn_map, decoder_state = model.decode(sess, encoder_output, decoder_input, decoder_states=decoder_state)
    attention.append(attn_map)
    token_highest_prob = np.argmax(decoder_output.flatten())
    if token_highest_prob == nlc_data.EOS_ID or len(output_sent) > FLAGS.max_seq_len:
      break
    output_sent += [token_highest_prob]
    decoder_input = np.array([token_highest_prob, ], dtype=np.int32).reshape([1, 1])

  return output_sent


def print_beam(beam, string='Beam'):
  print(string, len(beam))
  for (i, ray) in enumerate(beam):
    print(i, ray[0], detokenize(ray[2], reverse_vocab))
#    print(i, ray[0], ray[2])


def zip_input(beam):
  inp = np.array([ray[2][-1] for ray in beam], dtype=np.int32).reshape([1, -1])
  return inp


def zip_state(beam):
  if len(beam) == 1:
    return None # Init state
  return [np.array([(ray[1])[i, :] for ray in beam]) for i in xrange(FLAGS.num_layers)]


def unzip_state(state):
  beam_size = state[0].shape[0]
  return [np.array([s[i, :] for s in state]) for i in xrange(beam_size)]


def log_rebase(val):
  return np.log(10.0) * val


def lmscore(ray, v):
  if lm is None:
    return 0.0

  sent = ' '.join(ray[3])
  if len(sent) == 0:
    return 0.0

  if v == nlc_data.EOS_ID:
    return sum(w[0] for w in list(lm.full_scores(sent, eos=True))[-2:])
  elif reverse_vocab[v] in string.whitespace:
    return list(lm.full_scores(sent, eos=False))[-1][0]
  else:
    return 0.0


def beam_step(beam, candidates, decoder_output, zipped_state, max_beam_size):
  logprobs = (decoder_output).squeeze(axis=0) # [batch_size x vocab_size]
  newbeam = []

  for (b, ray) in enumerate(beam):
    prob, _, seq, low = ray
    for v in reversed(list(np.argsort(logprobs[b, :]))): # Try to look at high probabilities in each ray first

      newprob = prob + logprobs[b, v] + FLAGS.alpha * lmscore(ray, v)

      if reverse_vocab[v] in string.whitespace:
        newray = (newprob, zipped_state[b], seq + [v], low + [''])
      elif v >= len(nlc_data._START_VOCAB):
        newray = (newprob, zipped_state[b], seq + [v], low[:-1] + [low[-1] + reverse_vocab[v]])
      else:
        newray = (newprob, zipped_state[b], seq + [v], low)

      if len(newbeam) > max_beam_size and newprob < newbeam[0][0]:
        continue

      if v == nlc_data.EOS_ID:
        candidates += [newray]
        candidates.sort(key=lambda r: r[0])
        candidates = candidates[-max_beam_size:]
      else:
        newbeam += [newray]
        newbeam.sort(key=lambda r: r[0])
        newbeam = newbeam[-max_beam_size:]

  print('Candidates: %f - %f' % (candidates[0][0], candidates[-1][0]))
  print_beam(newbeam)
  return newbeam, candidates


def decode_beam(model, sess, encoder_output, max_beam_size):
  state, output = None, None
  beam = [(0.0, None, [nlc_data.SOS_ID], [''])] # (cumulative log prob, decoder state, [tokens seq], ['list', 'of', 'words'])

  candidates = []
  while True:
    output, attn_map, state = model.decode(sess, encoder_output, zip_input(beam), decoder_states=zip_state(beam))
    beam, candidates = beam_step(beam, candidates, output, unzip_state(state), max_beam_size)
    if beam[-1][0] < 1.5 * candidates[0][0]:
      # Best ray is worse than worst completed candidate. candidates[] cannot change after this.
      break

  print_beam(candidates, 'Candidates')
  finalray = candidates[-1]
  return finalray[2]


def fix_sent(model, sess, sent):
  # Tokenize
  input_toks, mask = tokenize(sent, vocab)
  # Encode
  encoder_output = model.encode(sess, input_toks, mask)
  # Decode
  output_toks = decode_beam(model, sess, encoder_output, FLAGS.beam_size)
  # De-tokenize
  output_sent = detokenize(output_toks, reverse_vocab)
  # Return
  return output_sent

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
