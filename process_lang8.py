import random
import argparse
from os.path import join as pjoin

'''
convert lang8 corpus into 2 text files containing parallel verb phrases
(original and corrected).

about half the examples don't have corrections; can either ignore or add those as identity mappings

currently doesn't do additional preprocessing besides that done by tajiri et al.
'''

def parse_entries(lines):
    par_count = 0
    original = list()
    corrected = list()
    for line in lines:
        cols = line.split('\t')
        if len(cols) < 6:
            continue
        else:
            # NOTE prints examples with multiple corrections
            #if len(cols) > 6:
                #print(' ||| '.join(cols[4:]))
            # NOTE not using multiple corrections to avoid train and dev having same source sentences
            for corr_sent in cols[5:6]:
                if cols[4] == corr_sent:
                    continue
                original.append(cols[4])
                corrected.append(corr_sent)
                par_count += 1
    print('%d parallel examples' % par_count)
    return original, corrected

def write_text(split, data, part, lang8_path):
    with open(pjoin(lang8_path, 'entries.%s.%s' % (split, part)), 'w') as fout:
        fout.write('\n'.join(data))

def process_data(lang8_path, split, dev_split_fract):
    with open(pjoin(lang8_path, 'entries.%s' % split)) as fin:
        lines = fin.read().strip().split('\n')
        print('%d lines total in %s split' % (len(lines), args.split))
        original, corrected = parse_entries(lines)

    # shuffle
    random.seed(1234)
    combined = zip(original, corrected)
    random.shuffle(combined)
    original, corrected = zip(*combined)

    if split == 'train':
        ntrain = int(len(combined) * (1 - dev_split_fract))
        train_original, train_corrected = original[:ntrain], corrected[:ntrain]
        dev_original, dev_corrected = original[ntrain:], corrected[ntrain:]
        print('writing %d training examples, %d dev examples' %\
                (ntrain, len(combined) - ntrain))
        write_text('train', train_original, 'original', lang8_path)
        write_text('train', train_corrected, 'corrected', lang8_path)
        write_text('dev', dev_original, 'original', lang8_path)
        write_text('dev', dev_corrected, 'corrected', lang8_path)
    else:
        write_text('test', original, 'original', lang8_path)
        write_text('test', corrected, 'corrected', lang8_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang8_path', help='path to directory containing lang-8 entries.[split] files')
    parser.add_argument('split', type=str, help='split to use, train refers to entries.train before creating dev split', choices=['train', 'test'])
    parser.add_argument('--dev_split_fract', default=0.01, type=float, help='fraction of training data to use for dev, split (e.g. 0.01)')
    args = parser.parse_args()

    process_data(args.lang8_path, args.split, args.dev_split_fract)

if __name__ == '__main__':
    main()
