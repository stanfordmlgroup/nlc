import sys
import subprocess
from os.path import join as pjoin

KENLM_PATH = "~/libs/kenlm/"
QUERY_BINARY = pjoin(KENLM_PATH, "bin/query")
LINES = 1000

lm_file = sys.argv[1]
input_file1 = sys.argv[2]
input_file2 = sys.argv[3]

def compute_sentence_scores(f):
  cmd = "%s -v sentence %s < %s | head -n %d" % (QUERY_BINARY, lm_file, f, LINES)
  print(cmd)
  output = subprocess.check_output(cmd, shell=True)
  print(output)
  scores = [float(s.split()[1]) for s in output.strip().split('\n')]
  return scores

scores1 = compute_sentence_scores(input_file1)
scores2 = compute_sentence_scores(input_file2)

higher = sum([scores1[k] > scores2[k] for k in xrange(len(scores1))])
print("%d/%d sentences in %s score higher than in %s" % (higher, len(scores1), input_file1, input_file2))
