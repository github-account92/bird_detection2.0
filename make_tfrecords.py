import argparse
import os
import sys

import librosa
import numpy as np
import tensorflow as tf

from data_utils import make_dev_inds, make_labeled_data_list, \
    prepare_transform, read_data_config, random_pad


def fulfill_config(config_path):
    """Checks whether the data for a requested config exists and creates it otherwise."""
    config_dict = read_data_config(config_path)

    train_path = config_dict["tfr_path"] + "_train.tfrecords"
    dev_path = config_dict["tfr_path"] + "_dev.tfrecords"
    if not os.path.exists(train_path) and not os.path.exists(dev_path):
        create_data_dir = input("The requested TFRecords files do not seem to "
                                "exist. Do you want to create them? This "
                                "could take a very long time. Type y/n (no "
                                "exits the program):")

        if create_data_dir.lower()[0] == "y":
            data_list = make_labeled_data_list(
                config_dict["data_dir"], config_dict["datasets"],
                config_dict["n_max"])
            transform = prepare_transform(config_dict)

            if not os.path.exists(config_dict["dev_inds"]):
                print("No dev inds found at specified path. Creating them...")
                make_dev_inds(data_list, 0.85, config_dict["dev_inds"])

            make_tfrecords(data_list, config_dict["tfr_path"],
                           dev_inds=np.load(config_dict["dev_inds"]),
                           resample_rate=config_dict["resample_rate"],
                           n_augment=config_dict["n_augment"],
                           transform=transform)

        else:
            sys.exit("TFRecords file does not exist and creation not "
                     "requested.")

    elif not os.path.exists(train_path) or not os.path.exists(dev_path):
        sys.exit("Either only training or dev file already exists. This is not"
                 " intended!")


def make_tfrecords(data_list, out_path, dev_inds, resample_rate=None,
                   n_augment=0, transform=None):
    """Consume an iterator and put everything into .tfrecords files.

    Parameters:
        data_list: Should return pairs of filename, label.
        out_path: Base path to store the resulting files to.
        dev_inds: np.array of ints. These will be taken as indices for the
                  heldout set.*
        resample_rate: Optional int giving the Hz to resample the data to.
        n_augment: Number of extra sequences to generate by mixing existing 
                   ones.
        transform: Transformation function to apply to the raw sequences. If
                   None, nothing is applied. This will only receive a sequence
                   as input, so prepare it accordingly beforehand.
    """
    def serialize(_seq, _label, writer):
        tfex = tf.train.Example(features=tf.train.Features(
            feature={"seq": tf.train.Feature(
                float_list=tf.train.FloatList(value=_seq.flatten())),
                "shape": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=_seq.shape)),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[_label]))}))
        writer.write(tfex.SerializeToString())

    with tf.python_io.TFRecordWriter(
                    out_path + "_train.tfrecords") as train_writer, \
            tf.python_io.TFRecordWriter(
                        out_path + "_dev.tfrecords") as dev_writer:
        for ind, (filename, label) in enumerate(data_list):
            seq, sr = librosa.load(filename, sr=resample_rate)
            if len(seq) / sr > 20:  # skip sequences > 20 seconds
                continue
            if transform:
                seq = transform(seq)
            else:  # raw: Add fake height/channel axis
                seq = seq[None, :]

            if ind in dev_inds:
                serialize(seq, label, dev_writer)
            else:
                serialize(seq, label, train_writer)
            if (ind + 1) % 100 == 0:
                print("Processed {} sequences!".format(ind+1))
    print("All done!")

    if n_augment:
        print("Augmenting with {} examples...".format(n_augment))
        # first some preparations...
        all_inds = np.arange(len(data_list))
        valid_inds = np.delete(all_inds, dev_inds)
        bird_files = [file for ind, (file, label) in enumerate(data_list)
                      if ind in valid_inds and label]
        noise_files = [file for ind, (file, label) in enumerate(data_list)
                       if ind in valid_inds and not label]

        prop_birds = len(bird_files) / (len(bird_files) + len(noise_files))
        p_pick_birds = 1 - np.sqrt(1 - prop_birds)

        with tf.python_io.TFRecordWriter(
                        out_path + "_augment.tfrecords") as aug_writer:
            for ind in range(n_augment):
                has_bird = False
                if np.random.rand() < p_pick_birds:
                    f1 = np.random.choice(bird_files)
                    has_bird = True
                else:
                    f1 = np.random.choice(noise_files)
                if np.random.rand() < p_pick_birds:
                    f2 = np.random.choice(bird_files)
                    has_bird = True
                else:
                    f2 = np.random.choice(noise_files)
                seq1, sr = librosa.load(f1, sr=resample_rate)
                seq2, sr = librosa.load(f2, sr=resample_rate)

                if len(seq1) / sr > 20 or len(seq2) / sr > 20:
                    continue

                longer = max(len(seq1), len(seq2))
                seq1 = random_pad(seq1, longer)
                seq2 = random_pad(seq2, longer)
                mixed_seq = 0.5 * (seq1 + seq2)

                if transform:
                    seq = transform(mixed_seq)
                else:  # raw: Add fake freq axis
                    seq = mixed_seq[None, :]
                serialize(seq, int(has_bird), aug_writer)

                if (ind + 1) % 100 == 0:
                    print("Generated {} sequences!".format(ind+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a TFRecords file for a given data config.")
    parser.add_argument("config_path",
                        help="Path to a data config file.")
    args = parser.parse_args()
    fulfill_config(args.config_path)
