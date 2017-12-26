import tensorflow as tf


def input_fn(data_path, subset, batch_size, freqs):
    """Builds an input function for tf.estimator.
    
    Parameters:
        data_path: Base path to tfrecords data file.
        subset: One of "train" or "dev".
        batch_size: Duh.
        freqs: Size of the frequency axis we can expect.
    
    Returns:
        get_next op of iterator.
    """
    data = tf.data.TFRecordDataset(data_path + "_" + subset + ".tfrecords")
    if subset == "train":
        data = data.shuffle(buffer_size=1000)
    data = data.map(parse_example)
    data = data.padded_batch(batch_size, ((1, freqs, -1), (1,)))
    if subset == "train":
        data = data.repeat()
    iterator = data.make_one_shot_iterator()
    return iterator.get_next()


def parse_example(example_proto):
    """Parse examples from a TFRecords file.

    Parameters:
        example_proto: The thing to parse.

    Returns: 
        The parsed thing. Note: This is always channels_first!
    """
    # TODO look into better ways of doing this
    # (SequenceExample sucks, maybe try bytes) (compare speeds!!)
    features = {"seq": tf.VarLenFeature(tf.float32),
                "shape": tf.FixedLenFeature((2,), tf.int64),
                "label": tf.FixedLenFeature((1,), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    sparse_seq = parsed_features["seq"]
    shape = tf.cast(parsed_features["shape"], tf.int32)
    dense_seq = tf.reshape(tf.sparse_to_dense(
        sparse_seq.indices, sparse_seq.dense_shape, sparse_seq.values),
                           shape)
    # add fake channel axis in any case
    return tf.expand_dims(dense_seq, axis=0), \
        tf.cast(parsed_features["label"], tf.int32)
