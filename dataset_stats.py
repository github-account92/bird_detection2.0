import argparse
import os

import librosa

from data_processing import make_labeled_iterator, make_unlabeled_iterator


def dataset_stats(iterator, filter_length):
    total_seqs = 0
    total_pos = 0
    total_length = 0
    min_length = float("inf")
    max_length = 0
    too_short = 0
    too_long = 0

    for filename, label in iterator:
        seq, _ = librosa.load(filename, sr=None)
        l = len(seq)
        if l < min_length:
            min_length = l
        if l > max_length:
            max_length = l
        if filter_length:
            if l > 600000:
                too_long += 1
                print("Skipping {} with length {}".format(filename, l))
                continue
            if l < 120000:
                too_short += 1
                print("Skipping {} with length {}".format(filename, l))
                continue

        total_pos += label
        total_seqs += 1
        total_length += len(seq) / 44100

    print("Total: {} Positive: {} Average: {}".format(total_seqs, total_pos,
                                                      total_pos/total_seqs))
    print("Total length: Seconds: {} Minutes {}: Hours: {}".format(
        total_length, total_length/60, total_length/3600))
    print("Longest sequence: {} Shortest: {}".format(max_length, min_length))
    print("Skipped {} too long sequences and {} too short ones".format(
        too_long, too_short))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path",
                        help="Path to where the data is.")
    parser.add_argument("-f", "--filter", action="store_true")
    parser.add_argument("-t", "--transform", default="raw",
                        help="How to transform the input sequences. 'raw' "
                             "(default), 'stft' or 'mel'.")
    args = parser.parse_args()

    print("\nStats for both training sets...")
    ITER = make_labeled_iterator(args.data_path, ["freefield", "warblr"])
    dataset_stats(ITER, filter_length=args.filter)

    print("\nStats for freefield training set...")
    ITER = make_labeled_iterator(args.data_path, ["freefield"])
    dataset_stats(ITER, filter_length=args.filter)

    print("\nStats for warblr training set...")
    ITER = make_labeled_iterator(args.data_path, ["warblr"])
    dataset_stats(ITER, filter_length=args.filter)

    print("\nStats for test set...")
    ITER = make_unlabeled_iterator(os.path.join(args.data_path, "testset"))
    dataset_stats(ITER, filter_length=False)
