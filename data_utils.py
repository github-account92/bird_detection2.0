import itertools
import os


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
