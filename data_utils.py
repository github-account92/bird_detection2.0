import itertools
import os

import librosa
import numpy as np


def make_labeled_data_list(base_path, sets, n_max=0):
    """Create a list of wave files with labels.

    Parameters:
        base_path: Folder that contains dataset folders as well as
                   corresponding labels.
        sets: Names of the datasets to include.
        n_max: Maximum number of files to include *per dataset*. If not given,
               will be size of each dataset.

    Returns: 
        List of len(sets)*n_max many files (or all files if n_max not 
        given) with their labels (tuples).
    """
    maps = [extract_labels(base_path, dataset) for dataset in sets]
    if n_max:
        maps = [m[:n_max] for m in maps]
    return list(itertools.chain(*maps))


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


def make_unlabeled_data_list(wav_dir):
    """Make a data list for the test set with "dummy" labels.

    Parameters:
        wav_dir: Directory containing the .wav folders

    Returns:
        List of all files in the directory, with dummy labels (tuples).
    """
    return list(zip([os.path.join(wav_dir, file) for file in os.listdir(
        wav_dir)],
               itertools.repeat(-1)))


DATA_CONFIG_REQUIRED_ENTRIES = {"tfr_path", "data_dir", "datasets",
                                "data_type", "dev_inds"}
DATA_CONFIG_OPTIONAL_ENTRIES = {"resample_rate", "n_max", "n_augment"}
DATA_CONFIG_ALLOWED_ENTRIES = DATA_CONFIG_REQUIRED_ENTRIES.union(
    DATA_CONFIG_OPTIONAL_ENTRIES)

DATA_TYPE_ENTRIES = {"raw": {},
                     "stft": {"window_size", "hop_length"},
                     "mel": {"window_size", "hop_length", "mel_freqs"}}
for data_type, entries in DATA_TYPE_ENTRIES.items():
    DATA_CONFIG_ALLOWED_ENTRIES = DATA_CONFIG_ALLOWED_ENTRIES.union(entries)

TO_INT_ENTRIES = {"resample_rate", "n_max", "n_augment", "window_size",
                  "hop_length", "mel_freqs"}


def read_data_config(config_path):
    """Read a config file with information about the data.

    The file should be in csv format and contain the following entries:
        tfr_path: BASE path to train/dev TFRecord files.
        data_dir: Path to the data directory (containing folders freefield and
                  warblr and the label files).
        datasets: Which datasets to use. Can be the name(s) of any number of
                  available datasets, separated by commas.
        data_type: One of "raw", "stft" or "mel".
        dev_inds: Path to .npy array containing indices to put in the
                  development set.
    Maybe required depending on data_type:
        window_size: For STFT. Not needed if data_type is "raw".
        hop_length: For STFT. Not needed if data_type is "raw".
        mel_freqs: Only relevant if data_type is "mel".
    Optional:
        resample_rate: If given, the raw data is resampled to this rate.
        n_max: If given, only process this many sequences per dataset.
        n_augment: If given, look for another TFRecord file with augmented 
                   data; if not found, create one.

    Entries can be in any order. Missing required entries will result in a
    crash, as will any superfluous (unexpected) entries.

    Returns:
        Dict with config file entries.
    """
    config_dict = dict()
    with open(config_path) as data_config:
        for line in data_config:
            line = line.strip().split(",")
            if line[0] == "datasets":  # can have more than one "value"
                key = line[0]
                val = line[1:]
            else:
                key, val = line
                val = maybe_to_int(key, val)
            config_dict[key] = val

    found_entries = set(config_dict.keys())
    for f_entry in found_entries:
        if f_entry not in DATA_CONFIG_ALLOWED_ENTRIES:
            raise ValueError("Entry {} found in config file which should not "
                             "be there.".format(f_entry))

    for r_entry in DATA_CONFIG_REQUIRED_ENTRIES:
        if r_entry not in found_entries:
            raise ValueError("Entry {} expected in config file, but not "
                             "found.".format(r_entry))

    for d_entry in DATA_TYPE_ENTRIES[config_dict["data_type"]]:
        if d_entry not in found_entries:
            raise ValueError("Entry {} expected for data_type {}, but not "
                             "found.".format(d_entry,
                                             config_dict["data_type"]))

    for o_entry in DATA_CONFIG_OPTIONAL_ENTRIES:
        if o_entry not in found_entries:
            print("Optional entry {} not found in config file. Setting to "
                  "None!".format(o_entry))
            config_dict[o_entry] = None

    return config_dict


def prepare_transform(config_dict):
    """Prepare the appropriate function for a requested data transformation."""
    trans = config_dict["data_type"]
    if trans == "raw":
        return None
    elif trans == "stft":
        return lambda seq: np.log(np.abs(librosa.stft(
            seq, n_fft=config_dict["window_size"],
            hop_length=config_dict["hop_length"])))
    elif trans == "mel":
        return lambda seq: np.log(librosa.feature.melspectrogram(
            y=seq, sr=config_dict["resample_rate"] or 44100,
            n_fft=config_dict["window_size"],
            hop_length=config_dict["hop_length"],
            n_mels=config_dict["mel_freqs"]))


def make_dev_inds(data_list, prop_train, out_path):
    """Create a numpy array with indices for a development set.

    Parameters:
        data_list: List with data entries -- needed for its length.
        prop_train: Float, desired proportional size of the training set.
                    1-prop_train is the probability of each example being put
                    into the dev set.
        out_path: Path to store the array to.
    """
    dev_inds = []
    for ind, thing in enumerate(data_list):
        if np.random.rand() >= prop_train:
            dev_inds.append(ind)
    np.save(out_path, np.array(dev_inds))


def maybe_to_int(key, val):
    """Converts data config entries to int if needed."""
    if key in TO_INT_ENTRIES:
        return int(val)
    return val


def random_pad(seq, target_length):
    """Randomly pad numpy array in one axis.

    Parameters:
        seq: The array to pad.
        target_length: int, Desired length.

    Returns:
        Padded array.
    """
    diff = target_length - len(seq)
    if diff == 0:  # nothing to do
        return seq
    elif diff < 0:
        print("Warning!! Padding requested for sequence longer than the target"
              " length. Nothing is done! Sequence length: {} Target: "
              "{}".format(len(seq), target_length))
        return seq

    # otherwise pad
    part_front = np.random.randint(low=0, high=abs(diff)+1)
    part_back = abs(diff) - part_front
    return np.pad(seq, (part_front, part_back), mode="constant")
