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

import numpy as np
from six.moves import xrange
import tensorflow as tf

import nlc_model
import nlc_data

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.0, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 400, "Vocabulary size for word model.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")

FLAGS = tf.app.flags.FLAGS

class PairIter:
  def __init__(self, fnamex, fnamey, batch_size, num_layers):
    self.fdx, self.fdy = open(fnamex), open(fnamey)
    self.batch_size = batch_size
    self.num_layers = num_layers
    self.batches = []

  def __iter__(self):
    return self

  def refill(self):
    line_pairs = []
    linex, liney = self.fdx.readline(), self.fdy.readline()

    while linex and liney:
      line_pairs.append((linex, liney))
      if len(line_pairs) == self.batch_size * 16:
        break
      linex, liney = self.fdx.readline(), self.fdy.readline()

    line_pairs.sort(lambda x, y: len(x))

    for batch_start in xrange(0, len(line_pairs), int(len(line_pairs)/self.batch_size)):
      x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+self.batch_size])
      if len(x_batch) < self.batch_size:
        break
      self.batches.append((x_batch, y_batch))

  def next(self):
    if len(self.batches) == 0:
      self.refill()
    if len(self.batches) == 0:
      raise StopIteration()

    def tokenize(batch):
      return map(lambda string: [int(s) for s in string.split()], batch)

    def add_sos_eos(tokens):
      return map(lambda token_list: [nlc_data.SOS_ID] + token_list + [nlc_data.EOS_ID], tokens)

    def padded(tokens, depth):
      maxlen = max(map(lambda x: len(x), tokens))
      align = pow(2, depth - 1)
      padlen = maxlen + (align - maxlen) % align
      return map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens)

    x_batch, y_batch = self.batches.pop(0)
    x_tokens, y_tokens = tokenize(x_batch), tokenize(y_batch)
    y_tokens = add_sos_eos(y_tokens)
    x_padded, y_padded = padded(x_tokens, self.num_layers), padded(y_tokens, 1)

    source_tokens = np.array(x_padded).T
    source_mask = (source_tokens != nlc_data.PAD_ID).astype(np.int32)
    target_tokens = np.array(y_padded).T
    target_mask = (target_tokens != nlc_data.PAD_ID).astype(np.int32)

    return source_tokens, source_mask, target_tokens, target_mask

def create_model(session, forward_only):
  model = nlc_model.NLCModel(
      FLAGS.vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
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


def train():
  """Train a translation model using NLC data."""
  # Prepare NLC data.
  print("Preparing NLC data in %s" % FLAGS.data_dir)

  x_train, y_train, x_dev, y_dev, _, _ = nlc_data.prepare_nlc_data(
    FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.vocab_size, FLAGS.vocab_size,
    tokenizer=get_tokenizer(FLAGS))

  with tf.Session() as sess:
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    epoch = 0
    while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
      epoch += 1
      current_step = 0
      exp_cost = None
      exp_length = None
      exp_norm = None

      ## Train
      for source_tokens, source_mask, target_tokens, target_mask in PairIter(x_train, y_train, FLAGS.batch_size, FLAGS.num_layers):
        # Get a batch and make a step.
        tic = time.time()

        _, grad_norm, cost = model.train(sess, source_tokens, source_mask, target_tokens, target_mask)

        toc = time.time()
        iter_time = toc - tic
        current_step += 1

        lengths = np.sum(target_mask, axis=0)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        if not exp_cost:
          exp_cost = cost
          exp_length = mean_length
          exp_norm = grad_norm
        else:
          exp_cost = 0.99*exp_cost + 0.01*cost
          exp_length = 0.99*exp_length + 0.01*mean_length
          exp_norm = 0.99*exp_norm + 0.01*grad_norm

        cost = cost / mean_length

        if current_step % FLAGS.print_every == 0:
          print('epoch %d, iter %d, cost %f, exp_cost %f, grad_norm %f, batch time %f, length mean/std %f/%f' %
                (epoch, current_step, cost, exp_cost / exp_length, grad_norm, iter_time, mean_length, std_length))

      ## Checkpoint
      checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)

      valid_costs, valid_lengths = [], []
      for source_tokens, source_mask, target_tokens, target_mask in PairIter(x_dev, y_dev, FLAGS.batch_size, FLAGS.num_layers):
        cost, _ = model.test(sess, source_tokens, source_mask, target_tokens, target_mask)
        valid_costs.append(cost * target_mask.shape[1])
        valid_lengths.append(np.sum(target_mask[1:, :]))
      valid_cost = sum(valid_costs) / float(sum(valid_lengths))

      print("Epoch %d Validation cost: %f" % (epoch, valid_cost))

      previous_losses.append(valid_cost)
      if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        sess.run(model.learning_rate_decay_op)
      sys.stdout.flush()

def main(_):
  train()

if __name__ == "__main__":
  tf.app.run()
