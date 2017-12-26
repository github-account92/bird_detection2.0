import argparse

import tensorflow as tf

from input import input_fn
from models import model_fn


parser = argparse.ArgumentParser(
    description="Train a CNN to recognize bird song.")
parser.add_argument("data_path",
                    help="Base path to tfrecords data (without "
                         "train/dev/extension!).")
parser.add_argument("model_dir",
                    help="Path to store checkpoints etc.")

parser.add_argument("-a", "--act",
                    default="relu",
                    choices=["relu", "elu"],
                    help="Which activation function to use. Can be one of "
                         "'relu' (default) or 'elu'.")
parser.add_argument("-b", "--batch_norm",
                    action="store_true",
                    help="Set to use batch normalization.")

parser.add_argument("-A", "--adam_params",
                    nargs=2,
                    type=float,
                    default=[1e-3, 1e-8],
                    metavar=["adam_lr", "adam_eps"],
                    help="Learning rate and epsilon for Adam.")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=32,
                    help="Batch size. Default: 32.")
parser.add_argument("-E", "--eval_freq",
                    type=int,
                    default=1000,
                    help="How often to evaluate. Default: Every 1000 steps. "
                         "Doing this too often may result in significant "
                         "slowdown of the overall process.")
parser.add_argument("-F", "--format",
                    default="channels_first",
                    choices=["channels_first", "channels_last"],
                    help="Data format. Either 'channels_first' (default, "
                         "recommended for GPU) or 'channels_last', recommended"
                         " for CPU.")
parser.add_argument("-M", "--mode",
                    default="raw",
                    choices=["raw", "spectrum", "mel"],
                    help="What mode to run (what kind of input). Can be one of"
                         " 'raw' (default), 'spectrum' or 'mel'")
parser.add_argument("-S", "--steps",
                    type=int,
                    default=10000,
                    help="How many training steps to take. Default: 10000.")
parser.add_argument("-V", "--vis",
                    action="store_true",
                    help="If set, add more visualizations (inputs and filter "
                         "maps).")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

# setup
FREQS = 128
DATA_FORMAT = args.format
USE_BN = args.batch_norm
BATCH_SIZE = args.batch_size
ACT = args.act
TRAIN_STEPS = args.steps
EVAL_STEPS = args.eval_freq
LEARN_RATE = args.adam_params[0]
EPSILON = args.adam_params[1]

if ACT == "elu":
    act = tf.nn.elu
else:  # since no other choice is allowed
    act = tf.nn.relu


print("Defining model...")
n_filters = [25, 25, 25, 25]
size_filters = [3, 3, 5, 5]
strides_filters = [1, 1, 1, 1]
size_pools = [1, 2, 1, 2]
strides_pools = [1, 2, 1, 2]

params = {"conv": {"filters": n_filters,
                   "sizes": size_filters,
                   "strides": strides_filters},
          "pool": {"sizes": size_pools,
                   "strides": strides_pools,
                   "fun": tf.layers.max_pooling2d},
          "act": act,
          "use_bn": USE_BN,
          "adam": {"lr": LEARN_RATE,
                   "eps": EPSILON},
          "data_format": DATA_FORMAT,
          "vis": {"imgs": args.vis}}

estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir=args.model_dir)


def train_input_fn(): return input_fn(args.data_path, "train",
                                      freqs=FREQS, batch_size=BATCH_SIZE)


def eval_input_fn(): return input_fn(args.data_path, "dev",
                                     freqs=FREQS, batch_size=BATCH_SIZE)

# note: a logging hook for the cost as well as a step counter are created
# automatically
logging_hook = tf.train.LoggingTensorHook(
    {"eval/accuracy": "eval/batch_accuracy"},
    every_n_iter=100,
    at_end=True)
steps_taken = 0
while steps_taken < TRAIN_STEPS:
    estimator.train(input_fn=train_input_fn, steps=EVAL_STEPS,
                    hooks=[logging_hook])
    steps_taken += EVAL_STEPS
    estimator.evaluate(input_fn=eval_input_fn)
