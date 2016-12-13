## BPE

First apply `LANG=iso-8859-1 sed -i 's/[\d128-\d255]//g' FILENAME` to remove non-ASCII characters
from source and target. (`nlc_data.py` should also filter away non-ASCII characters,
but we don't want them in vocabulary and here we construct vocabulary ourselves.)

Then take half of source training, half of target training, concatenate together.
(This is to ensure backoff to unigram characters in each.)

Then learn vocabulary using `learn_bpe.py`, say it outputs `vocab.dat`.
Then apply `vocab.dat` to `train` and `valid` files you want to run with.

Then run `add_backoff_unigrams.py vocab.dat new_vocab.dat`.
Finally, add in these tokens at the top of `new_vocab.dat`:
  <pad>
  <sos>
  <eos>
  <unk>
and copy `new_vocab.dat` to data directory.
