# command line interface for running ASR models
import argparse
from est_main import run_birds


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    choices=["train", "predict", "eval", "return"],
                    help="What to do. 'train', 'predict', 'eval' or "
                         "'return' The latter simply returns the estimator "
                         "object.")
parser.add_argument("data_config",
                    help="Path to data config file. See code for details.")
parser.add_argument("model_config",
                    help="Path to model config file. See code for details.")
parser.add_argument("model_dir",
                    help="Path to store checkpoints etc.")

parser.add_argument("-a", "--act",
                    default="relu",
                    choices=["relu", "elu"],
                    help="Which activation function to use. "
                         "Can be one of 'relu' (default),"
                         "or 'elu'.")
parser.add_argument("-b", "--batchnorm",
                    action="store_true",
                    help="Set to use batch normalization.")

parser.add_argument("-A", "--adam_params",
                    nargs=4,
                    type=float,
                    # these are *not* the TF defaults!!
                    default=[1e-3, 0.9, 0.9, 1e-8],
                    metavar=["adam_lr", "adam_eps"],
                    help="Learning rate, beta1 and beta2 and epsilon for "
                         "Adam.")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=64,
                    help="Batch size. Default: 64.")
parser.add_argument("-C", "--clipping",
                    type=float,
                    default=0.0,
                    help="Global norm to clip gradients to. Default: 0 (no "
                         "clipping).")
parser.add_argument("-E", "--renorm",
                    action="store_true",
                    help="Use batch renormalization (only has an effect if "
                         "batch norm is used).")
parser.add_argument("-F", "--data_format",
                    default="channels_first",
                    choices=["channels_first", "channels_last"],
                    help="Data format. Either 'channels_first' "
                         "(default, recommended for GPU) "
                         "or 'channels_last', recommended for CPU.")
parser.add_argument("-G", "--augment",
                    action="store_true",
                    help="Use augmented training data. Will lead to a crash "
                         "if no such data is available!")
parser.add_argument("-L", "--label_smoothing",
                    type=float,
                    default=0.0,
                    help="Label smoothing to apply. Default: 0 (no "
                         "smoothing).")
parser.add_argument("-N", "--normalize",
                    action="store_true",
                    help="Normalize inputs to have mean 0 and variance 1.")
parser.add_argument("-O", "--onedim",
                    action="store_true",
                    help="Use 1D convolutions instead of 2D.")
parser.add_argument("-R", "--reg",
                    type=float,
                    default=0.0,
                    help="Regularizer coefficient. Default: 0 (no "
                         "regularization). Currently does nothing!!")
parser.add_argument("-S", "--steps",
                    type=int,
                    default=50000,
                    help="Number of training steps to take. Default: 50000. "
                         "Ignored if doing prediction or evaluation.")
parser.add_argument("-T", "--threshold",
                    action="store_true",
                    help="Threshold input: Remove anything 80db below max.")
parser.add_argument("-U", "--use_avg",
                    action="store_true",
                    help="Use average pooling over time at the end instead of "
                         "max pooling.")
parser.add_argument("-V", "--vis",
                    type=int,
                    default=100,
                    help="If set, add visualizations of gradient norms and "
                         "activation distributions as well as graph profiling."
                         " This number signifies per how many steps you want "
                         "to add summaries. Profiling is added this many steps"
                         " times 50 (e.g. every 5000 steps if this is set to "
                         "100). Default: 100. Setting this to 0 will only plot"
                         " curves for loss and steps per second, every 100 "
                         "steps. This may result in faster execution.")
args = parser.parse_args()


out = run_birds(mode=args.mode, data_config=args.data_config,
                model_config=args.model_config, model_dir=args.model_dir,
                act=args.act, batchnorm=args.batchnorm,
                adam_params=args.adam_params, augment=args.augment,
                batch_size=args.batch_size, clipping=args.clipping,
                data_format=args.data_format,
                label_smoothing=args.label_smoothing, normalize=args.normalize,
                onedim=args.onedim, reg=args.reg, renorm=args.renorm,
                steps=args.steps, threshold=args.threshold,
                use_avg=args.use_avg, vis=args.vis)
