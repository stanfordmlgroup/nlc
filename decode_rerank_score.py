import os
import numpy as np
from subprocess import check_call, check_output
from collections import defaultdict

ALPHAS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
VISIBLE_DEVICES = "0,1"
NLM_CKPT = "/deep/u/borisi/sdlg/lm/50k_train_logs/train/model.ckpt-355520"

SRC_FILE = "/deep/group/nlp_data/nlc_data/nlc_dev.x.txt"
TGT_FILE = "/deep/group/nlp_data/nlc_data/nlc_dev.y.txt"
# Downloaded from https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
MULTIBLEU = "multi-bleu.perl"

BEAM_SIZE = 8

HYPS_FILE = "hyps_beam%d.pk" % BEAM_SIZE
NGRAM_OUTPUTS_DIR = "ngram_beam%d" % BEAM_SIZE
NLM_OUTPUTS_DIR = "nlm_beam%d" % BEAM_SIZE
PLOT_FILE = "ngram_nlm_beam%d.pdf" % BEAM_SIZE

# FIXME Currently outputs encoder-decoder hypotheses and scores
# as well as beam search with n-gram result
def decode(hyps_file, outputs_dir, alphas):
  cmd = "CUDA_VISIBLE_DEVICES=%s python decode.py --data_dir /deep/group/nlp_data/nlc_data --train_dir /deep/u/avati/nlc/data --hpconfig num_layers=2 --vocab_path /deep/u/borisi/sdlg/lm/1b_word_vocab_50k.txt --input_file %s --lmfile /deep/group/zxie/lms/commoncrawl.5g.1f.binary --hyps_file %s --sents_dir %s --alphas %s --beam_size %d" % (VISIBLE_DEVICES, SRC_FILE, HYPS_FILE, NGRAM_OUTPUTS_DIR, ALPHAS, BEAM_SIZE)
  print(cmd)
  check_call(cmd, shell=True)

def rerank(nlm_ckpt, hyps_file, outputs_dir, alphas):
  cmd = "CUDA_VISIBLE_DEVICES=%s python rerank.py --data_dir /deep/group/nlp_data/nlc_data --train_dir /deep/u/avati/nlc/data --hpconfig num_layers=2 --vocab_path /deep/u/borisi/sdlg/lm/1b_word_vocab_50k.txt --lm1b_ckpt %s --hyps_file %s --sents_dir %s --alphas %s" % (VISIBLE_DEVICES, nlm_ckpt, hyps_file, outputs_dir, alphas)
  print(cmd)
  check_call(cmd, shell=True)

def score_and_plot(ngram_outputs_dir, nlm_outputs_dir, plot_file):
  scores = defaultdict(list)
  for d in [ngram_outputs_dir, nlm_outputs_dir]:
    for a in ALPHAS.split(','):
      sents_file = os.path.join(d, "alpha%s.txt" % a)
      cmd = "perl %s %s < %s" % (MULTIBLEU, sents_file, TGT_FILE)
      print(cmd)
      output = check_output(cmd, shell=True)
      score = float(output.split(',')[0].split(" = ")[1])
      scores[d].append(score)
  from plot_utils import setup_mpl_defaults, plot_line_graph, save_graph_pdf
  # TODO Turn back on for machines with LaTeX
  #setup_mpl_defaults()
  plot_line_graph([float(a) for a in ALPHAS.split(',')],
          scores.values(), scores.keys())
  print(plot_file)
  save_graph_pdf(plot_file)
  return scores

if __name__ == "__main__":
  # TODO Currently just calling shell scripts, but
  # eventually will want to import components and plug in
  # - encoder-decoder model
  # - language model
  # - scoring metrics
  # TODO argparse

  for d in (NGRAM_OUTPUTS_DIR, NLM_OUTPUTS_DIR):
    if not os.path.exists(d):
      os.makedirs(d)

  #decode(HYPS_FILE, NGRAM_OUTPUTS_DIR, ALPHAS)
  #rerank(NLM_CKPT, HYPS_FILE, NLM_OUTPUTS_DIR, ALPHAS)
  scores = score_and_plot(NGRAM_OUTPUTS_DIR, NLM_OUTPUTS_DIR, PLOT_FILE)
