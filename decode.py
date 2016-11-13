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

import nlc_model
import nlc_data

from language_model import LM
from common_custom import CheckpointLoader
from data_utils import Vocabulary, Dataset

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

# LM 1B configurations.
tf.app.flags.DEFINE_string("temp_str_path", "/tmp/encdec-tmp", "Where the file with one string will be located.")
tf.app.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of GPUs used.")
tf.app.flags.DEFINE_string("log_dir", "/tmp", "")
tf.app.flags.DEFINE_string("vocab_path", "/deep/group/nlp_data/lm1b/1b_word_vocab.txt", "Overrides default vocab file path.")
tf.app.flags.DEFINE_string("lm1b_ckpt", "", "ckpt file of a trained LM 1B model.")

FLAGS = tf.app.flags.FLAGS
reverse_vocab, vocab = None, None
lm, lm_vocab, lm_hps = None, None, None


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


def detokenize(sents, reverse_vocab):
  # TODO: char vs word
  def detok_sent(sent):
    outsent = ''
    for t in sent:
      if t >= len(nlc_data._START_VOCAB):
        outsent += reverse_vocab[t]
    return outsent
  return [detok_sent(s) for s in sents]


def get_lm_perplexity(s, sess):
  print(s)
  # Yeah this is impractical, but we first just want
  # to see results!
  with open(FLAGS.temp_str_path, 'w') as f:
    f.write(s)
  dataset = Dataset(lm_vocab, FLAGS.temp_str_path, deterministic=True) 
  data_iterator = dataset.iterate_once(lm_hps.batch_size * lm_hps.num_gpus, lm_hps.num_steps)
  loss_nom = 0.0
  loss_den = 0.0
  for i, (x, y, w) in enumerate(data_iterator):
    loss = sess.run(lm.loss, {lm.x: x, lm.y: y, lm.w: w})
    loss_nom += loss
    loss_den += w.mean()
    loss = loss_nom / loss_den
    print("%d: %.3f (%.3f) ... " % (i, loss, np.exp(loss)))
  
  log_perplexity = loss_nom / loss_den
  print("Results: log_perplexity = %.3f perplexity = %.3f" % (log_perplexity, np.exp(log_perplexity)))
  return np.exp(log_perplexity)


def lm_rank(strs, probs, sess):
  if lm is None:
    return strs[0]
  a = FLAGS.alpha
  # This is the only line that needs to be changed to accomodate a
  # new LM.
  lmscores = [ (-1.0 * get_lm_perplexity(s, sess))/(1+len(s.split())) for s in strs ]
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
  best_str = lm_rank(beam_strs, probs, sess)
  # Return
  return best_str

def decode():
  # Prepare NLC data.
  global reverse_vocab, vocab, lm, lm_vocab, lm_hps

  lm_hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
  lm_hps.num_gpus = FLAGS.num_gpus

  lm_vocab = Vocabulary.from_file(FLAGS.vocab_path)

  with tf.variable_scope("model"):
    lm_hps.num_sampled = 0  # Always using full softmax at evaluation.
    lm_hps.keep_prob = 1.0
    lm = LM(lm_hps, "eval", "/cpu:0")

  if lm_hps.average_params:
    print("Averaging parameters for evaluation.")
    saver = tf.train.Saver(lm.avg_dict)
  else:
    saver = tf.train.Saver()

  ckpt_loader = CheckpointLoader(saver, lm.global_step, FLAGS.log_dir + "/train", FLAGS.lm1b_ckpt)

  print("Preparing NLC data in %s" % FLAGS.data_dir)

  #x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
  #  FLAGS.data_dir + '/' + FLAGS.tokenizer.lower(), FLAGS.max_vocab_size,
  #  tokenizer=get_tokenizer(FLAGS))
  #vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path)
  #vocab_size = len(vocab)
  #print("Vocabulary size: %d" % vocab_size)

  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=20,
                          inter_op_parallelism_threads=1)
  sess = tf.Session(config=config)
  with sess.as_default():
    if not ckpt_loader.load_checkpoint():
      print("Error loading LM checkpoint!")
      return
    
    print("Loaded checkpoint " + repr(ckpt_loader.last_global_step))
    sess.run(tf.initialize_local_variables())
    
    # Quick sanity check
    #print(lm_rank(['this a sent ence is vewy bad', 'this sentence is great !'], [0.9, 0.1], sess))
    #return
    
    # Running LM 1B on the first 1000 sentences of train.
    top_n = 1000
    bad = [line.strip().rstrip('\n') for line in open('/deep/u/borisi/sdlg/nlc/char/train.x.txt', 'r')]
    good = [line.strip().rstrip('\n') for line in open('/deep/u/borisi/sdlg/nlc/char/train.y.txt', 'r')]
    ppls = list()
    for idx in xrange(len(good[:top_n])):
      ppl_pair = (get_lm_perplexity(good[idx], sess), get_lm_perplexity(bad[idx], sess))
      ppls.append(ppl_pair)
      if ppl_pair[0] > ppl_pair[1]:
        print("==PPL_NOTICE== Good had higher ppl than bad for GOOD: %.3f %s --- BAD: %.3f %s" % (ppl_pair[0], good[idx], ppl_pair[1], bad[idx]))
      # Make it write out after each sentence pair
      sys.stdout.flush()
 
    return

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
