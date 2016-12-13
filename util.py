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

import nlc_data
import numpy as np
from six.moves import xrange
import tensorflow as tf
import random


FLAGS = tf.app.flags.FLAGS

def tokenize(string):
  return [int(s) for s in string.split()]

def pair_iter(fnamex, fnamey, batch_size, num_layers, sort_and_shuffle=True):
  fdx, fdy = open(fnamex), open(fnamey)
  batches = []

  while True:
    if len(batches) == 0:
      refill(batches, fdx, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
    if len(batches) == 0:
      break

    x_tokens, y_tokens = batches.pop(0)
    y_tokens = add_sos_eos(y_tokens)
    x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

    source_tokens = np.array(x_padded).T
    source_mask = (source_tokens != nlc_data.PAD_ID).astype(np.int32)
    target_tokens = np.array(y_padded).T
    target_mask = (target_tokens != nlc_data.PAD_ID).astype(np.int32)

    yield (source_tokens, source_mask, target_tokens, target_mask)

  return

def refill(batches, fdx, fdy, batch_size, sort_and_shuffle=True):
  line_pairs = []
  linex, liney = fdx.readline(), fdy.readline()

  while linex and liney:
    x_tokens, y_tokens = tokenize(linex), tokenize(liney)

    if len(x_tokens) < FLAGS.max_seq_len and len(y_tokens) < FLAGS.max_seq_len:
      line_pairs.append((x_tokens, y_tokens))
    if len(line_pairs) == batch_size * 16:
      break
    linex, liney = fdx.readline(), fdy.readline()

  if sort_and_shuffle:
    line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

  for batch_start in xrange(0, len(line_pairs), batch_size):
    x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
#    if len(x_batch) < batch_size:
#      break
    batches.append((x_batch, y_batch))

  if sort_and_shuffle:
    random.shuffle(batches)
  return

def add_sos_eos(tokens):
  return map(lambda token_list: [nlc_data.SOS_ID] + token_list + [nlc_data.EOS_ID], tokens)

def padded(tokens, depth):
  maxlen = max(map(lambda x: len(x), tokens))
  align = pow(2, depth - 1)
  padlen = maxlen + (align - maxlen) % align
  return map(lambda token_list: token_list + [nlc_data.PAD_ID] * (padlen - len(token_list)), tokens)


def get_tokenizer(flags):
  if flags.tokenizer.lower() == 'bpe':
    return nlc_data.bpe_tokenizer
  elif flags.tokenizer.lower() == 'char':
    return nlc_data.char_tokenizer
  elif flags.tokenizer.lower() == 'word':
    return nlc_data.basic_tokenizer
  else:
    raise
  return tokenizer

