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

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import nlc_data

class GRUCellAttn(rnn_cell.GRUCell):
  def __init__(self, num_units, encoder_output, scope=None):
    self.hs = encoder_output
    with vs.variable_scope(scope or type(self).__name__):
      with vs.variable_scope("Attn1"):
        self.phi_hs = tanh(rnn_cell.linear(self.hs, num_units, True, 1.0))
    super(GRUCellAttn, self).__init__(num_units)

  def __call__(self, inputs, state, scope=None):
    gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
    with vs.variable_scope(scope or type(self).__name__):
      with vs.variable_scope("Attn2"):
        gamma_h = tanh(rnn_cell.linear(gru_out, self._num_units, True, 1.0))
      weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
      weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
      weights = weights / (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
      context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
      with vs.variable_scope("AttnConcat"):
        out = tf.relu(rnn_cell.linear([context, gru_out], self._num_units, True, 1.0))
      return (out, tf.slice(weights, [0, 0, 0], [-1, -1, 1]))

class NLCModel(object):
  def __init__(self, vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, forward_only=False):

    self.size = size
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    self.source_tokens = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="source_tokens")
    self.target_tokens = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="target_tokens")
    self.source_mask = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="source_mask")
    self.target_mask = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="target_mask")
    self.sequence_length = None

    self.setup_embeddings()
    self.setup_encoder()
    self.setup_decoder()
    self.setup_loss()

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)

      gradients = tf.gradients(self.losses, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norms = norm
      self.updates = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())

  def setup_embeddings(self):
    with vs.variable_scope("embeddings"):
      self.L_enc = tf.get_variable("L_enc", [self.vocab_size, self.size])
      self.L_dec = tf.get_variable("L_dec", [self.vocab_size, self.size])
      self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.source_tokens)
      self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)

  def setup_encoder(self):
    self.encoder_cell = rnn_cell.GRUCell(self.size)
    self.encoder_output, self.encoder_hidden = self.bidirectional_rnn(self.encoder_cell, self.encoder_inputs)

  def setup_decoder(self):
    self.decoder_cell = GRUCellAttn(self.size, self.encoder_hidden, scope=None)
    self.decoder_output, self.decoder_hidden = rnn.dynamic_rnn(self.decoder_cell, self.decoder_inputs, time_major=True,
                                                               dtype=dtypes.float32, sequence_length=self.sequence_length,
                                                               scope=None)
  def setup_loss(self):
    pass

  def bidirectional_rnn(self, cell, inputs, sequence_length=None):
    name = "BiRNN"
    # Forward direction
    with vs.variable_scope(name + "_FW") as fw_scope:
      output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=True, dtype=dtypes.float32,
                                                   sequence_length=sequence_length, scope=fw_scope)

    # Backward direction
    # TODO implement _reverse_seq on Tensor
#    with vs.variable_scope(name + "_BW") as bw_scope:
#      tmp, output_state_bw = rnn.dynamic_rnn(cell, rnn._reverse_seq(inputs, sequence_length),
#                                             time_major=True, dtype=dtypes.float32,
#                                             sequence_length=sequence_length, scope=bw_scope)
    output_bw = 0 #rnn._reverse_seq(tmp, sequence_length)

    outputs = output_fw + output_bw
    output_state = output_state_fw + 0 #output_state_bw

    return (outputs, output_state)

  def train(self, session, encoder_inputs, decoder_inputs, targets, target_weights):
    input_feed = {}
    input_feed["encoder"] = encoder_inputs
    input_feed["decoder"] = decoder_inputs
    input_feed["targets"] = targets
    input_feed["weights"] = target_weights

    output_feed = [self.updates, self.gradient_norms, self.losses]

    outputs = session.run(output_feed, input_feed)

    return outputs[1], outputs[2]

  def test(self, session, encoder_inputs, decoder_inputs, targets, target_weights):
    input_feed = {}
    input_feed["encoder"] = encoder_inputs
    input_feed["decoder"] = decoder_inputs
    input_feed["targets"] = targets
    input_feed["weights"] = target_weights

    output_feed = [self.losses, self.outputs]

    outputs = session.run(output_feed, input_feed)

    return outputs[0], outputs[1]
