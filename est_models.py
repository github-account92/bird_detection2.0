import tensorflow as tf

from hooks import SummarySaverHookWithProfile


def model_fn(features, labels, mode, params, config):
    """Model function for tf.estimator.

    Parameters:
        features: Should be a batch_size x channels x height x width tensor of
                  input sequences.
                  Note: Must be channels_first!!
        labels: batch_size tensor of class labels.
        mode: Train, Evaluate or Predict modes from tf.estimator.
        params: Should be a dict with the following string keys:
            model_config: Path to config file to build the model up to the
                          pre-final layer..
            vocab_size: Size of the vocabulary, to get the size for the final
                        layer.
            act: The activation function, e.g. tf.nn.relu or tf.nn.elu.
            use_bn: Bool, whether to use batch normalization.
            data_format: String, channels_first or otherwise assumed to be 
                         channels_last (this is not checked here!!).
            adam_args: List with Adam parameters (in order!!).
            clipping: Float, to set the gradient clipping norm. 0 disables 
                      clipping.
            vis: Int, whether to include visualizations besides loss and steps
                 per time and if so how often.
            reg: Float, coefficient for regularizer for conv layers. 0 disables
                 it.
            onedim: Use 1D convolution.
            label_smoothing: Float, amount of label smoothing to use (0 to 
                             disable)
            normalize: Normalize inputs to mean 0 and variance 1.
            renorm: Use batch renormalization.
            use_avg: Average-pool at the end instead of max-pool.
        config: RunConfig object passed through from Estimator.

    Returns:
        An EstimatorSpec to be used in tf.estimator.
    """
    # first get all the params
    model_config = params["model_config"]
    act = params["act"]
    use_bn = params["use_bn"]
    data_format = params["data_format"]
    adam_args = params["adam_args"]
    clipping = params["clipping"]
    vis = params["vis"]
    reg = params["reg"]
    onedim = params["onedim"]
    label_smoothing = params["label_smoothing"]
    normalize = params["normalize"]
    renorm = params["renorm"]
    use_avg = params["use_avg"]

    # model input -> output
    with tf.variable_scope("model"):
        if normalize:
            means, variances = tf.nn.moments(features, axes=[2, 3],
                                             keep_dims=True)
            features = (features - means) / variances

        if data_format == "channels_last":
            if onedim:  # b x 1 x t x 128
                features = tf.transpose(features, [0, 1, 3, 2])
            else:  # b x 128 x t x 1
                features = tf.transpose(features, [0, 2, 3, 1])
        elif onedim:  # channels_first but onedim b x 128 x 1 x t
            features = tf.transpose(features, [0, 2, 1, 3])
        # otherwise b x 1 x 128 x t

        pre_out, total_stride, all_layers = read_apply_model_config(
            model_config, features, act=act, batchnorm=use_bn,
            train=mode == tf.estimator.ModeKeys.TRAIN, data_format=data_format,
            vis=vis, reg=reg, onedim=onedim, renorm=renorm)
        reduce_fun = tf.reduce_mean if use_avg else tf.reduce_max
        reduce_over_time = reduce_fun(
            pre_out, axis=-1 if data_format == "channels_first" else -2,
            name="reduce_over_time")
        flattened = tf.layers.flatten(reduce_over_time, name="flattened")
        logits = tf.layers.dense(flattened, 1, activation=None, name="logits")
        if vis:
            tf.summary.histogram("logits", logits)

        # if in prediction mode, this is all we need
        predictions = {"logits": logits,
                       "probabilities": tf.nn.sigmoid(
                           logits, name="probabilities"),
                       "classes": tf.cast(
                           tf.greater_equal(logits, 0), tf.int32,
                           name="classes"),
                       "input": features,
                       "flattened": flattened}
        for name, act in all_layers:
            predictions[name] = act
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

    # loss
    cross_ent = tf.losses.sigmoid_cross_entropy(
        logits=logits, multi_class_labels=labels,
        label_smoothing=label_smoothing)
    with tf.variable_scope("eval", reuse=True):
        acc = tf.reduce_mean(
            tf.cast(tf.equal(labels, predictions["classes"]), tf.float32),
            name="batch_accuracy")
        tf.summary.scalar("accuracy", acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(*adam_args)
            if use_bn:
                # necessary for batchnorm to work properly in inference mode
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op, grads_and_vars, glob_grad_norm = clip_and_step(
                        optimizer, cross_ent, clipping)
            else:
                train_op, grads_and_vars, glob_grad_norm = clip_and_step(
                    optimizer, cross_ent, clipping)
        # visualize gradients
        if vis:
            with tf.name_scope("visualization"):
                for g, v in grads_and_vars:
                    if v.name.find("kernel") >= 0:
                        tf.summary.scalar(v.name + "gradient_norm", tf.norm(g))
                tf.summary.scalar("global_gradient_norm", glob_grad_norm)

        # The combined summary/profiling hook needs to be created in here
        scaff = tf.train.Scaffold()
        hooks = []
        if vis:
            save_and_profile = SummarySaverHookWithProfile(
                save_steps=vis, profile_steps=50*vis,
                output_dir=config.model_dir, scaffold=scaff)
            hooks.append(save_and_profile)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                          train_op=train_op, scaffold=scaff,
                                          training_hooks=hooks)

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


###############################################################################
# Helper functions for building inference models.
###############################################################################
def read_apply_model_config(config_path, inputs, act, batchnorm, train,
                            data_format, vis, reg, onedim, renorm):
    """Read a model config file and apply it to an input.

    A config file is a csv file where each line stands for a layer or a whole
    residual block. Lines should follow the format:
    type,n_f,w_f,s_f
        type: "layer", "block" or "dense", stating whether this is a single
              conv layer, a residual block or a dense block.
        n_f: Number of filters in the layer/block. For dense blocks, this is
             the growth rate!
        w_f: Size of filters.
        s_f: Convolutional stride of the layer/block.
    ALTERNATIVELY:
    pool,1,w_p,s_p for pooling, with size (w) and stride (s)

    NOTE: This is for 1/2D convolutional models.
          The final layer should *not* be included since it's always the same
          and depends on the data (i.e. vocabulary size).

    Parameters:
        config_path: Path to the model config file.
        inputs: 4D inputs to the model.
        act: Activation function to apply in each layer/block.
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add histograms for layer activations.
        reg: Float, coefficient for regularizer for conv layers. 0 disables it.
        onedim: Use 1D convolutions instead of 2D. NOTE that this is
                implemented via 2D convolutions; input needs to be in correct
                shape.
        renorm: Use batch renormalization.

    Returns:
        Output of the last layer/block, total stride of the network and a list
        of all layer/block activations with their names (tuples name, act).
    """
    # TODO for resnets/dense nets, return all *layers*, not just blocks

    def parse_line(l):
        entries = l.strip().split(",")
        return entries[0], int(entries[1]), int(entries[2]), int(entries[3])

    print("Reading, building and applying model...")
    total_pars = 0
    all_layers = []
    with open(config_path) as model_config:
        total_stride = 1
        previous = inputs
        for ind, line in enumerate(model_config):
            t, n_f, w_f, s_f = parse_line(line)
            name = t + str(ind)
            if t == "layer":
                previous, pars = conv_layer(
                    previous, n_f, w_f, s_f, act, batchnorm, train,
                    data_format, vis, name=name, reg=reg, onedim=onedim,
                    renorm=renorm)
            elif t == "pool":
                if onedim:
                    previous = tf.layers.max_pooling2d(
                        previous, (1, w_f), (1, s_f), padding="same",
                        data_format=data_format, name=name)
                else:
                    previous = tf.layers.max_pooling2d(
                        previous, w_f, s_f, padding="same",
                        data_format=data_format, name=name)
                pars = 0
            else:
                raise ValueError(
                    "Invalid layer type specified in layer {}! Valid are "
                    "'layer', 'pool'. You specified "
                    "{}.".format(ind, t))
            all_layers.append((name, previous))
            total_stride *= s_f
            total_pars += pars
    print("Number of model parameters: {}".format(total_pars))
    return previous, total_stride, all_layers


def conv_layer(inputs, n_filters, size_filters, stride_filters, act,
               batchnorm, train, data_format, vis, name, reg, onedim,
               renorm):
    """Build and apply a 1D/2D convolutional layer.

    Parameters:
        inputs: 3D/4D inputs to the layer.
        n_filters: Number of filters for the layer.
        size_filters: Filter width for the layer.
        stride_filters: Stride for the layer.
        act: Activation function to apply after convolution (or optionally
             batch normalization).
        batchnorm: Bool, whether to use batch normalization.
        train: Bool or TF placeholder. Fed straight into batch normalization
               (ignored if that is not used).
        data_format: channels_first or _last. Assumed that you checked validity
                     beforehand. I.e. if it's not first, this function simply
                     assumes that it's last.
        vis: Bool, whether to add a histogram for layer activations.
        name: Name of the layer (used for variable scope and summary).
        reg: Coefficient for regularizer; currently unused.
        onedim: Use 1D convolutions instead of 2D. NOTE that this is
                implemented via 2D convolutions; input needs to be in correct
                shape.
        renorm: Use batch renormalization

    Returns:
        Output of the layer and number of parameters.
    """
    channel_axis = 1 if data_format == "channels_first" else -1
    n_pars = int(inputs.shape[channel_axis]) * n_filters * size_filters
    if not onedim:
        n_pars *= size_filters

    if batchnorm:  # add per-filter beta and gamma
        n_pars += 2 * n_filters
    print("\tCreating layer {} with {} parameters...".format(name, n_pars))

    with tf.variable_scope(name):
        if onedim:
            conv = tf.layers.conv2d(
                inputs, filters=n_filters, kernel_size=(1, size_filters),
                strides=(1, stride_filters),
                activation=None if batchnorm else act,
                use_bias=not batchnorm, padding="same",
                data_format=data_format, name="conv")
        else:
            conv = tf.layers.conv2d(
                inputs, filters=n_filters, kernel_size=size_filters,
                strides=stride_filters,
                activation=None if batchnorm else act,
                use_bias=not batchnorm, padding="same",
                data_format=data_format, name="conv")
        if batchnorm:
            conv = tf.layers.batch_normalization(
                conv, axis=channel_axis, training=train, name="batch_norm",
                renorm=renorm)
            if act:
                conv = act(conv)
        if vis:
            tf.summary.histogram("activations_" + name,
                                 conv)
        return conv, n_pars


###############################################################################
# Helper functions for various purposes.
###############################################################################
def clip_and_step(optimizer, loss, clipping):
    """Helper to compute/apply gradients with clipping.

    Parameters:
        optimizer: Subclass of tf.train.Optimizer (e.g. GradientDescent or
                   Adam).
        loss: Scalar loss tensor.
        clipping: Threshold to use for clipping.

    Returns:
        The train op.
        List of gradient, variable tuples, where gradients have been clipped.
        Global norm before clipping.
    """
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, varis = zip(*grads_and_vars)
    if clipping:
        grads, global_norm = tf.clip_by_global_norm(grads, clipping,
                                                    name="gradient_clipping")
    else:
        global_norm = tf.global_norm(grads, name="gradient_norm")
    grads_and_vars = list(zip(grads, varis))  # list call is apparently vital!!
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=tf.train.get_global_step(),
        name="train_step")
    return train_op, grads_and_vars, global_norm
