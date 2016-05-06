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

import numpy as np
from six.moves import xrange
import tensorflow as tf

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
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")

FLAGS = tf.app.flags.FLAGS

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

def tokenize(sent, vocab, depth):
  align = pow(2, depth - 1)
  token_ids = nlc_data.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS))
  ones = [1] * len(token_ids)
  pad = (align - len(token_ids)) % align

  token_ids += [nlc_data.PAD_ID] * pad
  ones += [0] * pad

  source = np.array(token_ids).reshape([-1, 1])
  mask = np.array(ones).reshape([-1, 1])

  return source, mask

def decode():
  # Prepare NLC data.
  print("Preparing NLC data in %s" % FLAGS.data_dir)

  x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
    FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
    tokenizer=get_tokenizer(FLAGS))
  vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path)
  vocab_size = len(vocab)
  print("Vocabulary size: %d" % len(vocab))

  with tf.Session() as sess:
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)

    while True:
      sent = raw_input("Enter a sentence: ")
      source, mask = tokenize(sent, vocab, FLAGS.num_layers)

      encoder_output = model.encode(sess, source, mask)
      decoder_state = None
      decoder_input = np.array([nlc_data.SOS_ID, ], dtype=np.int32).reshape([1, 1])
      attention = []
      output_sent = ''
      while True:
        decoder_output, attn_map, decoder_state = model.decode(sess, encoder_output, decoder_input, decoder_states=decoder_state)
        attention.append(attn_map)
        token_highest_prob = np.argmax(decoder_output.flatten())
        if token_highest_prob == nlc_data.EOS_ID or len(output_sent) > 200:
          break
        output_sent += reverse_vocab[token_highest_prob]
        decoder_input = np.array([token_highest_prob, ], dtype=np.int32).reshape([1, 1])

      print("Candidate: ", ''.join(output_sent))

def main(_):
  decode()

if __name__ == "__main__":
  tf.app.run()
