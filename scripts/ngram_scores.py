import sys
import subprocess
from os.path import join as pjoin

KENLM_PATH = "~/libs/kenlm/"
QUERY_BINARY = pjoin(KENLM_PATH, "bin/query")
LINES = 400

lm_file = sys.argv[1]
input_file1 = sys.argv[2]
input_file2 = sys.argv[3]

def compute_sentence_scores(f):
  cmd = "cat %s | perl /deep/group/nlp_data/nlc_data/tokenizer.perl -q -no-escape | %s -v sentence %s | head -n %d" % (f, QUERY_BINARY, lm_file, LINES)
  print cmd
  output = subprocess.check_output(cmd, shell=True)
  print(output)
  sents = open(f, 'r').read().strip().split('\n')
  scores = [float(s.split()[1]) for s in output.strip().split('\n')]
  return scores, sents

scores1, sents1 = compute_sentence_scores(input_file1)
scores2, sents2 = compute_sentence_scores(input_file2)

print("Mistakes:")
for score1, sent1, score2, sent2 in zip(scores1, sents1, scores2, sents2):
  if score1 > score2:
    print(sent1 + " | " + sent2)

higher = sum([scores1[k] > scores2[k] for k in xrange(len(scores1))])
print("%d/%d sentences in %s score higher than in %s" % (higher, len(scores1), input_file1, input_file2))
