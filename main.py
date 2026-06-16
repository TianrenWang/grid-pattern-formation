import numpy as np
import torch.cuda
import argparse


from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir",
    # default='/mnt/fs2/bsorsch/grid_cells/models/',
    default="output/",
    help="directory to save trained models",
)
parser.add_argument(
    "--n_epochs", default=100, help="number of training epochs", type=int
)
parser.add_argument("--n_steps", default=1000, help="batches per epoch", type=int)
parser.add_argument(
    "--batch_size", default=200, help="number of trajectories per batch", type=int
)
parser.add_argument(
    "--sequence_length", default=20, help="number of steps in trajectory", type=int
)
parser.add_argument(
    "--learning_rate", default=1e-4, help="gradient descent learning rate", type=float
)
parser.add_argument("--Np", default=512, help="number of place cells", type=int)
parser.add_argument("--Ng", default=4096, help="number of grid cells", type=int)
parser.add_argument(
    "--place_cell_rf",
    default=0.12,
    help="width of place cell center tuning curve (m)",
    type=float,
)
parser.add_argument(
    "--surround_scale",
    default=2,
    help="if DoG, ratio of sigma2^2 to sigma1^2",
    type=int,
)
parser.add_argument("--RNN_type", default="RNN", help="RNN or LSTM")
parser.add_argument("--activation", default="relu", help="recurrent nonlinearity")
parser.add_argument(
    "--weight_decay",
    default=1e-4,
    help="strength of weight decay on recurrent weights",
    type=float,
)
parser.add_argument(
    "--DoG", default=True, help="use difference of gaussians tuning curves"
)
parser.add_argument(
    "--periodic", default=False, help="trajectories with periodic boundary conditions"
)
parser.add_argument(
    "--box_width", default=2.2, help="width of training environment", type=float
)
parser.add_argument(
    "--box_height", default=2.2, help="height of training environment", type=float
)
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device to use for training",
)
parser.add_argument(
    "--seed", default=None, help="seed number for all numpy random number generator"
)

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f"Using device: {options.device}")

if options.seed:
    np.random.seed(int(options.seed))

place_cells = PlaceCells(options)
if options.RNN_type == "RNN":
    model = RNN(options, place_cells)
elif options.RNN_type == "LSTM":
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
if options.device == "cuda":
    print("Using CUDA")
model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
