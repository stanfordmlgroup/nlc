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
from tensorflow.python.ops import variable_scope as vs

import nlc_data

class NLCModel(object):
  def __init__(self, vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, forward_only=False):

    self.size = size
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # Use MultiRNNCell for now. (Later: change inner loop to be along timestep dimension, makes pyramid easy)
    cell = tf.nn.rnn_cell.GRUCell(size)
    self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    self.encoder_tokens = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="encoder_tokens")
    self.decoder_tokens = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="decoder_tokens")
    self.targets        = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="targets")
    self.target_weights = tf.placeholder(tf.int32, shape=[None, self.batch_size], name="weights")

    self.setup_embeddings()

    self.outputs, self.losses = self.create_output_losses(forward_only)

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
      self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.encoder_tokens)
      self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.decoder_tokens)

  def create_output_losses(self, forward_only):
    all_inputs = [self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights]
    with ops.op_scope(all_inputs, None, "seq2seq_dynamic_model"):
      outputs, _ = self.enc_dec_attn(forward_only)

      losses = tf.nn.seq2seq.sequence_loss(outputs[-1], self.targets, self.target_weights,
                                           average_across_batch=True,
                                           average_across_timesteps=False)
      return outputs, losses


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

  def enc_dec_attn(self, forward_only):
    with vs.variable_scope("embedding_attention_seq2seq"):
      # Encoder.
      encoder_outputs, encoder_state = self.bidirectional_rnn(self.cell, self.encoder_inputs)

      # Attention.
      attention_states = tf.transpose(encoder_outputs, perm=[1, 0, 2]) # [batch_size x TIME x dim]

      # Decoder.
      output_size = self.vocab_size
      decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(self.cell, output_size)

      return tf.nn.seq2seq.embedding_attention_decoder(self.decoder_inputs, encoder_state, attention_states,
                                                       decoder_cell, self.vocab_size, self.size, num_heads=1,
                                                       output_size=output_size, output_projection=None,
                                                       feed_previous=forward_only,
                                                       initial_state_attention=False)

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
