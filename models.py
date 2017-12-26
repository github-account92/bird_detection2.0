import tensorflow as tf

from utils import img_grid, GRIDS


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator.

    Parameters:
    features: Should be a batch_size x channels x height x width tensor of
              input sequences.
              Note: Must be channels_first!!
    labels: batch_size tensor of class labels.
    mode: Train, Evaluate or Predict modes from tf.estimator.
    params: Should be a dict with the following keys (strings):
        conv: Another dict, storing lists for the convolutional filters, filter
              sizes and strides for each layer.
              Can be integers or 2-tuples.
        pool: Another dict storing pooling sizes and strides in a similar
              manner, as well as the pooling function to use.
              NOTE: If a layer is not supposed to use pooling, use INTEGER 1
              for both sizes and strides for that layer.
        act: The activation function, e.g. tf.nn.relu or tf.nn.elu.
        use_bn: Bool, whether to use batch normalization.
        data_format: String, channels_first or otherwise assumed to be 
                     channels_last.

    Returns:
    An EstimatorSpec to be used in tf.estimator.
    """
    # first get all the params
    filters = params["conv"]["filters"]
    f_sizes = params["conv"]["sizes"]
    f_strides = params["conv"]["strides"]
    p_sizes = params["pool"]["sizes"]
    p_strides = params["pool"]["strides"]
    p_fun = params["pool"]["fun"]
    act = params["act"]
    use_bn = params["use_bn"]
    data_format = params["data_format"]
    learn_rate = params["adam"]["lr"]
    eps = params["adam"]["eps"]
    vis_imgs = params["vis"]["imgs"]

    # model input -> output
    with tf.variable_scope("model"):
        if data_format == "channels_last":
            features = tf.transpose(features, [0, 2, 3, 1])
        conved = conv_sequence(features, filters, f_sizes, f_strides, p_sizes,
                               p_strides, act, p_fun, use_bn,
                               mode == tf.estimator.ModeKeys.TRAIN,
                               data_format, vis_imgs)
        maxed_over_time = tf.reduce_max(
            conved, axis=3 if data_format == "channels_first" else 2,
            name="max_over_time")
        flattened = tf.layers.flatten(maxed_over_time, name="flattened")
        logits = tf.layers.dense(flattened, 1, activation=None, name="logits")

        # if in prediction mode, this is all we need
        predictions = {"logits": logits,
                       "probabilities": tf.nn.sigmoid(
                           logits, name="probabilities"),
                       "classes": tf.cast(
                           tf.greater_equal(logits, 0), tf.int32,
                           name="classes")}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

    # loss
    cross_ent = tf.losses.sigmoid_cross_entropy(logits=logits,
                                                multi_class_labels=labels)
    with tf.variable_scope("eval", reuse=True):
        acc = tf.reduce_mean(tf.cast(tf.equal(labels,
                                              predictions["classes"]),
                                     tf.float32),
                             name="batch_accuracy")
        tf.summary.scalar("accuracy", acc)

    # visualizations
    if vis_imgs:
        with tf.variable_scope("visualization", reuse=True):
            if data_format == "channels_first":
                tf.summary.image("inputs", tf.transpose(features,
                                                        [0, 2, 3, 1]))
            else:
                tf.summary.image("inputs", features)

    # rest of the setup for training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                           epsilon=eps)
        if use_bn:  # properly set up batchnorm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optimizer.compute_gradients(cross_ent)
                train_step = optimizer.apply_gradients(
                    grads_and_vars, global_step=tf.train.get_global_step(),
                    name="train_step")
        else:
            grads_and_vars = optimizer.compute_gradients(cross_ent)
            train_step = optimizer.apply_gradients(
                grads_and_vars, global_step=tf.train.get_global_step(),
                name="train_step")
        # visualize gradients
        with tf.variable_scope("visualization", reuse=True):
            for g, v in grads_and_vars:
                if v.name.find("kernel") >= 0:
                    tf.summary.scalar(v.name + "gradient_norm", tf.norm(g))
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                          train_op=train_step)

    # if we make it here, we're in evaluation mode
    with tf.variable_scope("eval", reuse=True):
        eval_metric_ops = {"eval/accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"],
            name="accuracy"),
                           "eval/precision": tf.metrics.precision(
                               labels=labels,
                               predictions=predictions["classes"],
                               name="precision"),
                           "eval/recall": tf.metrics.recall(
                               labels=labels,
                               predictions=predictions["classes"],
                               name="recall"),
                           "eval/auc": tf.metrics.auc(
                               labels=labels,
                               predictions=predictions["probabilities"],
                               name="auc")}
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                      eval_metric_ops=eval_metric_ops)


def conv_sequence(inputs, n_filters, size_filters, stride_filters, size_pools,
                  stride_pools, act, pool, batchnorm, train, data_format,
                  vis_fmaps):
    """Build and apply a sequence of convolutional layers.
    
    Parameters:
        inputs: 4D inputs to the first layer.
        n_filters, size_filters, stride_filters, size_pools, stride_pools:
                   Lists of integers or length-2 tuples of integers. Should all
                   be the same length.
                   NOTE: If some layer should not have pooling, both size and
                   strides should be 1 there. Just ints, not tuples!
        act: Activation function to apply after each layer.
        pool: Pooling function to use. Should be some kind of 2D pooling.
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis_fmaps: Bool, whether to add visualizations of feature maps.
                 
    Returns:
        Output of final convolutional layer.
    """
    print("Building {} convolutional layers...".format(len(n_filters)))

    channel_axis = 1 if data_format == "channels_first" else -1
    prev_layer = inputs
    for layer in range(len(n_filters)):
        with tf.variable_scope("layer" + str(layer)):
            conv = tf.layers.conv2d(prev_layer, n_filters[layer],
                                    size_filters[layer],
                                    strides=stride_filters[layer],
                                    activation=None if batchnorm else act,
                                    use_bias=not batchnorm,
                                    padding="same",
                                    data_format=data_format,
                                    name="conv")
            if batchnorm:
                conv = tf.layers.batch_normalization(conv, axis=channel_axis,
                                                     training=train, name="bn")
                conv = act(conv)
            if size_pools[layer] != 1 or stride_pools[layer] != 1:
                prev_layer = pool(conv, size_pools[layer], stride_pools[layer],
                                  padding="same", data_format=data_format,
                                  name="pool")
            else:
                prev_layer = conv

        # visualize feature maps
        if vis_fmaps:
            with tf.variable_scope("visualization", reuse=True):
                fmap_pics = [tf.expand_dims(t, axis=-1) for
                             t in tf.unstack(prev_layer, axis=channel_axis)]
                grid = img_grid(fmap_pics, *GRIDS[n_filters[layer]])
                tf.summary.image("fmaps_layer_" + str(layer), grid)
    return prev_layer
