import argparse
import itertools
import os

import librosa
import numpy as np
import tensorflow as tf


def make_tfrecords(iterator, out_path, prop_train=0.85, dev_inds=None,
                   resample_rate=None, transform="raw", n_augment=0):
    """Consume an iterator and put everything into .tfrecords files.
    
    Parameters:
        iterator: Should return pairs of filename, label.
        out_path: Base path to store the resulting files to.
        prop_train: Float, giving what proportion of the data is to be used for
                    training. The remaining data will be used as a dev set.
                    Ignored if dev_inds given!
        dev_inds: Optional path to a stored numpy.array of ints, if given these
                  will be taken indices for the holdout set. If this is given 
                  prop_train will be ignored! **Not implemented yet**
        resample_rate: Optional int giving the Hz to resample the data to.
        transform: What to do with the raw sequences. "raw", "stft" or "mel".
        n_augment: Huhu.
    """
    if dev_inds is not None:
        input("You have specified dev_inds, but this doesn't do anything yet."
              " Sure you want to continue?")

    if resample_rate:
        sr = resample_rate
    else:
        sr = 44100
    with tf.python_io.TFRecordWriter(
                    out_path + "_train.tfrecords") as train_writer, \
            tf.python_io.TFRecordWriter(
                        out_path + "_dev.tfrecords") as dev_writer:
        for ind, (filename, label) in enumerate(iterator):
            seq, _ = librosa.load(filename, sr=resample_rate)
            # TODO this will break with resampling
            if len(seq) > 600000 or len(seq) < 120000:
                continue
            if transform == "mel":
                seq = np.log(librosa.feature.melspectrogram(
                    seq, sr=sr, n_fft=5000, hop_length=1250))
            elif transform == "stft:":
                raise NotImplementedError("STFT is not implemented yet!")
            else:  # raw: Add fake freq axis
                seq = seq[None, :]

            tfex = tf.train.Example(features=tf.train.Features(
                feature={"seq": tf.train.Feature(
                    float_list=tf.train.FloatList(value=seq.flatten())),
                         "shape": tf.train.Feature(
                             int64_list=tf.train.Int64List(value=seq.shape)),
                         "label": tf.train.Feature(
                             int64_list=tf.train.Int64List(value=[label]))}))
            if np.random.rand() <= prop_train:
                train_writer.write(tfex.SerializeToString())
            else:
                dev_writer.write(tfex.SerializeToString())
            if (ind + 1) % 100 == 0:
                print("Processed {} sequences!".format(ind+1))


def make_labeled_iterator(base_path, sets, n_max=0):
    """Create an iterator over wave files.
    
    Parameters:
        base_path: Folder that contains dataset folders as well as
                   corresponding labels.
        sets: Names of the datasets to include.
        n_max: Maximum number of files to include *per dataset*. If not given,
               will be size of each dataset.
    
    Returns: 
        Iterator over len(sets)*n_max many files (or all files if n_max not 
        given) with their labels (tuples).
    """
    maps = [extract_labels(base_path, dataset) for dataset in sets]
    if n_max:
        maps = [m[:n_max] for m in maps]
    return itertools.chain(*maps)


def extract_labels(base_path, dataset):
    """Given a file of labels, return a list of corresponding tuples.
    
    Parameters:
        base_path: Folder that contains dataset folders as well as 
                   corresponding labels.
        dataset: Name of the dataset.
    
    Returns: 
        List of tuples (no dict to guarantee ordering) file_path, label
    """
    labels_path = os.path.join(base_path, "labels_" + dataset)
    with open(labels_path, mode="r") as labels_raw:
        label_map = []  # list of tuples instead of dict to guarantee ordering
        next(labels_raw)  # skip csv header
        for line in labels_raw:
            filename, label = line.split(",")
            label_map.append((os.path.join(
                base_path, dataset, filename + ".wav"),
                              int(label)))
    return label_map


def make_unlabeled_iterator(wav_dir):
    """Make an iterator for the test set with "dummy" labels.

    Parameters:
        wav_dir: Directory containing the .wav folders

    Returns:
        Iterator over all files in the directory, with dummy labels (tuples).
    """
    return zip([os.path.join(wav_dir, file) for file in os.listdir(wav_dir)],
               itertools.repeat(-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path",
                        help="Path to where the data is.")
    parser.add_argument("out_path",
                        help="Base path to output data files to.")
    parser.add_argument("-n", "--nseqs", type=int, default=0,
                        help="Number of sequences to use (per dataset!). If "
                             "not given, all sequences are processed.")
    parser.add_argument("-t", "--transform", default="raw",
                        help="How to transform the input sequences. 'raw' "
                             "(default), 'stft' or 'mel'.")
    args = parser.parse_args()

    ITER = make_labeled_iterator(args.data_path, ["freefield", "warblr"],
                                 n_max=args.nseqs)
    make_tfrecords(ITER, args.out_path, transform=args.transform)
