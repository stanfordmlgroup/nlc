import sys
from string import printable

vocab_file = sys.argv[1]
new_vocab_file = sys.argv[2]

with open(vocab_file, 'r') as fin:
  vocab = fin.read().strip().split('\n')

vocab_set = set(vocab)
# FIXME(zxie) Exclude whitespace for now
for c in printable[:-6]:
  if c not in vocab_set:
    print("Adding printable char %s" % c)
    vocab.append(c)

with open(new_vocab_file, 'w') as fout:
  fout.write('\n'.join(vocab))
