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
import pickle
from collections import defaultdict

import numpy as np
from six.moves import xrange
import tensorflow as tf

from language_model import LM
from common_custom import CheckpointLoader
from data_utils import Vocabulary, Dataset

# LM 1B configurations.
tf.app.flags.DEFINE_string("temp_str_path", "/tmp/encdec-tmp", "Where the file with one string will be located.")
tf.app.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of GPUs used.")
tf.app.flags.DEFINE_string("log_dir", "/tmp", "")
tf.app.flags.DEFINE_string("vocab_path", "1b_word_vocab.txt", "Overrides default vocab file path.")
tf.app.flags.DEFINE_string("lm1b_ckpt", "", "ckpt file of a trained LM 1B model.")

tf.app.flags.DEFINE_string("hyps_file", "", "Pickle file containing hyps and probabilities")
tf.app.flags.DEFINE_string("sents_dir", None, "Directory hypotheses and probabilities to")
tf.app.flags.DEFINE_string("alphas", "0.3,0.4", "Comma-separated anguage model relative weights to try.")

FLAGS = tf.app.flags.FLAGS
lm, lm_vocab, lm_hps = None, None, None

def get_lm_prob(s, sess):
  #print(s)
  # Yeah this is impractical, but we first just want
  # to see results!
  with open(FLAGS.temp_str_path, 'w') as f:
    # FIXME(zxie) Why does it sometimes output non UTF-8 characters?
    try:
      f.write(s.decode("utf-8"))
    except:
      # Tiny number
      return 1e-10
  dataset = Dataset(lm_vocab, FLAGS.temp_str_path, deterministic=True)
  data_iterator = dataset.iterate_once(lm_hps.batch_size * lm_hps.num_gpus, lm_hps.num_steps)
  loss_nom = 0.0
  loss_den = 0.0
  for i, (x, y, w) in enumerate(data_iterator):
    loss = sess.run(lm.loss, {lm.x: x, lm.y: y, lm.w: w})
    loss_nom += loss
    # NOTE Confusing line, due to multiplying by w mask then averaging over unmasked size
    # when computing the loss
    loss_den += w.mean()
    loss = loss_nom / loss_den
    #print("%d: %.3f (%.3f) ... " % (i, loss, np.exp(loss)))

  # This should already be normalized by length
  log_perplexity = loss_nom / loss_den
  log_prob = -log_perplexity
  return np.exp(log_prob)

def lm_rank(strs, probs, sess):
  if lm is None:
    return strs[0]
  # This is the only line that needs to be changed to accomodate a
  # new LM.
  # Using log10 to match kenlm
  lmscores = [np.log10(get_lm_prob(s, sess)) for s in strs]
  probs = [ p / (len(s)+1) for (s, p) in zip(strs, probs) ]
  #print("lm scores and probs")
  #print(lmscores)
  #print(probs)

  #for (s, p, l) in zip(strs, probs, lmscores):
    #print(s, p, l)

  alphas = [float(s) for s in FLAGS.alphas.split(",")]
  sents = list()
  for a in alphas:
    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    sents.append(generated)
  #return generated #, score, nw_score, lm_score
  return sents, FLAGS.alphas.split(",")

def decode():
  global lm, lm_vocab, lm_hps

  hyps_list, probs_list = pickle.load(open(FLAGS.hyps_file, "rb"))

  lm_hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
  lm_hps.num_gpus = FLAGS.num_gpus

  lm_vocab = Vocabulary.from_file(FLAGS.vocab_path)

  with tf.variable_scope("model"):
    lm_hps.num_sampled = 0  # Always using full softmax at evaluation.
    lm_hps.keep_prob = 1.0
    lm = LM(lm_hps, "eval")

  if lm_hps.average_params:
    print("Averaging parameters for evaluation.")
    saver = tf.train.Saver(lm.avg_dict)
  else:
    saver = tf.train.Saver()

  ckpt_loader = CheckpointLoader(saver, lm.global_step, FLAGS.log_dir + "/train", FLAGS.lm1b_ckpt)
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=20,
                          inter_op_parallelism_threads=1)
  sess = tf.Session(config=config)

  alpha_sents = defaultdict(list)
  with sess.as_default():
    if not ckpt_loader.load_checkpoint():
      print("Error loading LM checkpoint!")
      return
    print("Loaded checkpoint " + repr(ckpt_loader.last_global_step))
    sess.run(tf.initialize_local_variables())
    for hyps, probs in zip(hyps_list, probs_list):
      sents, alphas = lm_rank(hyps, probs, sess)
      for alpha, sent in zip(alphas, sents):
        alpha_sents[alpha].append(sent)

  for alpha in alpha_sents:
    with open(os.path.join(FLAGS.sents_dir, "alpha%s.txt" % alpha), "w") as fout:
      fout.write('\n'.join(alpha_sents[alpha]))

def main(_):
  decode()

if __name__ == "__main__":
  tf.app.run()
