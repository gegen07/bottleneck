from argparse import ArgumentParser
from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP
from distutils.util import strtobool

def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.NEIGHBORS_MATCH, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--type", dest="type", default=GNN_TYPE.GCN, type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)
    parser.add_argument("--dim", dest="dim", default=32, type=int, required=False)
    parser.add_argument("--depth", dest="depth", default=3, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=None, type=int, required=False)
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=50000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TRAIN, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument('--last_layer_fully_adjacent', action='store_true')
    parser.add_argument('--no_layer_norm', action='store_true')
    parser.add_argument('--no_activation', action='store_true')
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--unroll', action='store_true', help='use the same weights across GNN layers')

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--sheaf_decay', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--lr_decay_patience', type=int, default=20)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='loss')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh", help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)
    parser.add_argument('--use_bias', dest='use_bias', type=str2bool, default=True)

    # Experiment parameters
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, choices=['DiagSheaf', 'BundleSheaf', 'GeneralSheaf', 'DiagSheafODE',
                                                      'BundleSheafODE', 'GeneralSheafODE','CoopSheaf',
                                                      'MLP', 'GCN', 'GAT', 'SAGE'], default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--evectors', type=int, default=0, help="Number of Laplacian PE eigenvectors to use.")

    # ODE args
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")
    parser.add_argument('--int_method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")
    parser.add_argument("--num_bundles", type=int, default=1,)
    parser.add_argument("--diffusion_time", type=float, default=1,)
    parser.add_argument("--gnn_layers", type=int, default=0,)
    parser.add_argument("--gnn_hidden", type=int, default=16,)
    parser.add_argument("--pe_size", type=int, default=1,)

    args = parser.parse_args()
    Experiment(args).run()


def get_fake_args(
        task=Task.NEIGHBORS_MATCH,
        type=GNN_TYPE.GCN,
        dim=32,
        depth=3,
        num_layers=None,
        train_fraction=0.8,
        max_epochs=50000,
        eval_every=100,
        batch_size=1024,
        accum_grad=1,
        patience=20,
        stop=STOP.TRAIN,
        loader_workers=0,
        last_layer_fully_adjacent=False,
        no_layer_norm=False,
        no_activation=False,
        no_residual=False,
        unroll=False,
):
    return AttrDict({
        'task': task,
        'type': type,
        'dim': dim,
        'depth': depth,
        'num_layers': num_layers,
        'train_fraction': train_fraction,
        'max_epochs': max_epochs,
        'eval_every': eval_every,
        'batch_size': batch_size,
        'accum_grad': accum_grad,
        'stop': stop,
        'patience': patience,
        'loader_workers': loader_workers,
        'last_layer_fully_adjacent': last_layer_fully_adjacent,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
        'd': 2,
        'layers': 1,
        'gnn_layers': num_layers,
        'gnn_hidden': dim,
        'hidden_channels': dim,
        'left_weights': True,
        'right_weights': True,
        'orth': 'householder',
        'pe_size': 0,
        'sheaf_act': 'tanh',
        'input_dropout': 0.0,
        'dropout': 0.0,
        'use_act': False,
    })
