#! /usr/bin/env python3
#

"""The script for family tree or general graphs experiments."""
import warnings
import wandb
from torch.utils.tensorboard import SummaryWriter
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
from jactorch.optim.quickaccess import get_optimizer
from jactorch.optim.accum_grad import AccumGrad
from jactorch.data.dataloader import JacDataLoader
from jacinle.utils.meter import GroupMeters
from jacinle.utils.container import GView
from jacinle.logging import get_logger, set_output_file
from jacinle.cli.argument import JacArgumentParser
from difflogic.train import TrainerBase
from difflogic.thutils import binary_accuracy
from difflogic.nn.rl.reinforce import REINFORCELoss
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference, SparseLogicMachine
from sparse_hypergraph import SparseHypergraph
from difflogic.dataset.graph import SingleContrastiveLinkPredDataset
from difflogic.cli import format_args
import jactorch.nn as jacnn
import jacinle.io as io
import jacinle.random as random
import random as sys_random
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import json
import os
import functools
import collections
import copy
import time
from icecream import ic


def empty_cache():
    with torch.cuda.device('cuda:0'):
        torch.cuda.empty_cache()


def warn(*args, **kwargs):
    pass


warnings.warn = warn


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# TASKS = [
#     'outdegree', 'connectivity', 'adjacent', 'adjacent-mnist',
#     'has-father', 'has-sister', 'grandparents', 'uncle', 'family-of-three', 'three-generations',
#     'maternal-great-uncle', 'transitivity', 'self', 'family-self', 'ogbl-ddi', 'ogbl-collab'
# ]

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'sparse_nlm'],
    help='model choices, nlm: Neural Logic Machine')

parser.add_argument('--builtin-head', action='store_true',
                    help='use builtin head')
parser.add_argument('--benchmark', action='store_true',
                    help='do benchmarking')
parser.add_argument('--verbose', action='store_true',
                    help='print model')
parser.add_argument('--num-workers', type=int, default=0)
# NLM parameters, works when model is 'nlm'
nlm_group = parser.add_argument_group('Neural Logic Machines')
SparseLogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 4,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

nlm_group.add_argument(
    '--nlm-sparse-loss',
    type=str,
    default='hoyer_square'
)

# task related
task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, help='tasks choices')
task_group.add_argument(
    '--train-number',
    type=int,
    default=10,
    metavar='N',
    help='size of training instances')
task_group.add_argument(
    '--adjacent-pred-colors', type=int, default=4, metavar='N')
task_group.add_argument(
    '--num-rels', type=int, default=1, metavar='N')
task_group.add_argument('--outdegree-n', type=int, default=2, metavar='N')
task_group.add_argument(
    '--connectivity-dist-limit', type=int, default=4, metavar='N')

task_group.add_argument(
    '--transitivity-steps', type=int, default=2, metavar='N')
task_group.add_argument(
    '--transitivity-paths', type=int, default=1, metavar='N')

task_group.add_argument(
    '--task-is-directed', action='store_true', help='task is directed')

task_group.add_argument(
    '--arity', type=int, default=2, metavar='N')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-graph-method',
    default='edge',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.0,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-colors',
    type=int,
    default=4,
    metavar='N',
    help='number of colors in adjacent task')
data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')

data_gen_group.add_argument(
    '--aug-node-feature', type=str, default=None, help='augment node feature')
data_gen_group.add_argument(
    '--aug-node-feature-dim', type=int, default=0, help='augment node feature dims')

train_group = parser.add_argument_group('Train')
train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--multi-gpu', action='store_true', help='use multi-GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=1.0,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient for batches (default: 1)')
train_group.add_argument(
    '--ohem-size',
    type=int,
    default=0,
    metavar='N',
    help='size of online hard negative mining')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for training')
train_group.add_argument(
    '--subgraph',
    type=str,
    default='full',
    help='subgraph sampler to use')
train_group.add_argument(
    '--subgraph-size',
    type=int,
    default=1,
    metavar='N',
    help='subgraph size')
train_group.add_argument(
    '--IS-k',
    type=int,
    default=2,
    metavar='N',
    help='k in Information sufficiency')
train_group.add_argument(
    '--IS-gamma',
    type=float,
    default=1,
    metavar='F',
    help='gamma to adjust IS')
train_group.add_argument(
    '--resample',
    type=int,
    default=50,
    metavar='N',
    help='sample how many subgraphs in one large graph')
train_group.add_argument(
    '--do-calibrate', action='store_true',
    help='calibrate the label')
train_group.add_argument(
    '--walk-length',
    type=int,
    default=4,
    metavar='N',
    help='walk length')
train_group.add_argument(
    '--start-num',
    type=int,
    default=4,
    metavar='N',
    help='start num')
train_group.add_argument(
    '--neighbor-sizes',
    type=str,
    default='2_2',
    help='neighbor sizes')
train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')
train_group.add_argument(
    '--early-stop-loss-thresh',
    type=float,
    default=-1,
    metavar='F',
    help='threshold of loss for early stop')
train_group.add_argument(
    '--sparsity-loss-ratio',
    type=float,
    default=0,
    metavar='F',
    help='Sparsity loss ratio')
train_group.add_argument(
    '--label-smoothing',
    type=float,
    default=0.0,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--test-subgraph',
    type=str,
    default='full',
    help='subgraph sampler to use in testing')
train_group.add_argument(
    '--test-subgraph-size',
    type=int,
    default=100,
    metavar='N',
    help='subgraph size in testing')
train_group.add_argument(
    '--test-walk-length',
    type=int,
    default=4,
    metavar='N',
    help='walk length in testing')
train_group.add_argument(
    '--test-start-num',
    type=int,
    default=4,
    metavar='N',
    help='start num in testing')
train_group.add_argument(
    '--test-neighbor-sizes',
    type=str,
    default='2_2',
    help='neighbor sizes in testing')
train_group.add_argument(
    '--link-pred-k',
    type=int,
    default=2,
    help='single link pred dataset k')
train_group.add_argument(
    '--negative-sampling', type=float, default=0, help='negative sampling ratio')
train_group.add_argument(
    '--single-link-pred-bridge',
    type=str,
    default='rand',
    help='single link pred bridge')

train_group.add_argument(
    '--ranking-loss',
    action='store_true')
train_group.add_argument(
    '--contrastive-loss',
    action='store_true')
train_group.add_argument(
    '--ranking-margin',
    type=float,
    default=1
)
train_group.add_argument(
    '--train-num-neg-per-pos',
    type=int,
    default=1
)
train_group.add_argument(
    '--test-num-neg-per-pos',
    type=int,
    default=50
)
train_group.add_argument(
    '--node-label',
    action='store_true')

# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 50,
        'epoch_size': 250,
        'test_epoch_size': 20,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
    })

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', type=str, default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
io_group.add_argument(
    '--plot-dir', type=str, default=None, metavar='DIR', help='plot dir')
logger = get_logger(__file__)

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()


if args.plot_dir is not None:
    io.mkdir(args.plot_dir)

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)
else:
    args.checkpoints_dir = None
    args.summary_file = None


def set_seed(seed):
    sys_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if args.seed is not None:
    set_seed(args.seed)
    random.reset_global_seed(args.seed)
else:
    args.seed = random.seed()

if args.task[:4] == 'tsd-':
    args.task = args.task[4:]
    args.family_tree_format = 'tsd'
else:
    args.family_tree_format = 'fmsd'

args.task_is_outdegree = args.task in ['outdegree']
args.task_is_connectivity = args.task in ['connectivity']
args.task_is_transitivity = args.task in ['transitivity']
args.task_is_self = args.task == 'self'
args.task_is_adjacent = args.task in ['adjacent', 'adjacent-mnist']
args.task_is_family_tree = args.task in [
    'has-father', 'has-sister', 'grandparents', 'uncle', 'maternal-great-uncle', 'family-self', 'family-of-three', 'three-generations'
]
args.task_is_mnist_input = args.task in ['adjacent-mnist']
args.task_is_1d_output = args.task in [
    'outdegree', 'adjacent', 'adjacent-mnist', 'has-father', 'has-sister'
]
args.task_is_3d_output = args.task in ['family-of-three', 'three-generations']
args.task_is_link_prediction = 'grail-' in args.task or 'ogbl-' in args.task or 'kg-' in args.task or 'hkg-' in args.task
args.task_has_ternary = args.family_tree_format == 'tsd'
args.nlm_disable_mask = [int(i) for i in filter(
    len, args.nlm_disable_mask.split('_'))]

args.neighbor_sizes = [int(i) for i in filter(
    len, args.neighbor_sizes.split('_'))]


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = jacnn.Conv2dLayer(
            1, 10, kernel_size=5, batch_norm=True, activation='relu')
        self.conv2 = jacnn.Conv2dLayer(
            10,
            20,
            kernel_size=5,
            batch_norm=True,
            dropout=False,
            activation='relu')
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def nlm_monitor_update(monitors, name, info):
    for depth, l in enumerate(info):
        for order, v in enumerate(l):
            if torch.is_tensor(v):
                monitors.update(
                    {f'{name}-{depth+1}-{order}': v.detach().cpu()})
            else:
                monitors.update({f'{name}-{depth+1}-{order}': v})


class Model(nn.Module):
    """The model for family tree or general graphs path tasks."""

    def __init__(self):
        super().__init__()

        # inputs
        input_dim = 4 if args.task_is_family_tree else 1
        input_dim = 2 if args.family_tree_format == 'tsd' else input_dim
        self.feature_axis = 1 if args.task_is_1d_output else 2
        self.feature_axis = 3 if args.task_is_3d_output else self.feature_axis
        target_dim = args.adjacent_pred_colors if args.task_is_adjacent else 1

        input_dim = args.num_rels if args.task_is_link_prediction else input_dim
        target_dim = args.num_rels if args.task_is_link_prediction else target_dim

        # features
        if args.model == 'nlm':
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            if args.task_is_adjacent:
                input_dims[1] = args.gen_graph_colors
                if args.task_is_mnist_input:
                    self.lenet = LeNet()
            input_dims[args.arity] = input_dim
            if args.family_tree_format == 'tsd':
                input_dims[3] = 1
            self.features = LogicMachine.from_args(
                input_dims, args.nlm_attributes, args, prefix='nlm')
            output_dim = self.features.output_dims[self.feature_axis]
        elif args.model == 'sparse_nlm':
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            if args.task_is_adjacent:
                input_dims[1] = args.gen_graph_colors
                if args.task_is_mnist_input:
                    self.lenet = LeNet()
            input_dims[1] += args.aug_node_feature_dim
            input_dims[args.arity] = input_dim
            if args.family_tree_format == 'tsd':
                input_dims[3] = 1
            self.features = SparseLogicMachine.from_args(
                input_dims, args.nlm_attributes, args, prefix='nlm')
            output_dim = self.features.output_dims[self.feature_axis]

        norm = None if args.ranking_loss or args.contrastive_loss else 'sigmoid'
        if not args.builtin_head:
            self.pred = LogicInference(output_dim, target_dim, [], norm=norm)
        elif target_dim > 1:
            raise NotImplementedError

        if args.aug_node_feature_dim > 0:
            n_dict = {'ogbl-ddi': 4267,
                      'ogbl-collab': 235868, 'ogbl-biokg': 93773}
            self.aug_node_feature = nn.Embedding(
                n_dict[args.task], args.aug_node_feature_dim)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        states, relations, ternaries = None, None, None
        if hasattr(feed_dict, 'relations') and torch.is_tensor(feed_dict.relations):
            relations = feed_dict.relations.float()
        # ic(relations.shape)
        # ic(relations.nonzero())

        ret_dict = {}

        run_time, used_mem = 0, 0
        if args.model == 'nlm':
            inp = [None for _ in range(args.nlm_breadth + 1)]
            inp[1] = states  # None
            inp[2] = relations
            inp[3] = ternaries
            depth = None
            if args.benchmark:
                empty_cache()
                torch.cuda.reset_peak_memory_stats()
                mem_start = torch.cuda.memory_stats(
                )["active_bytes.all.peak"] // 1000000
                # print('Memory start: {}'.format(mem_start))
                torch.cuda.synchronize()
                time_start = time.time()
            ret_dict = self.features(inp, depth=depth, plot_dir=args.plot_dir)
            if args.benchmark:
                torch.cuda.synchronize()
                run_time = time.time() - time_start
                mem_end = torch.cuda.memory_stats(
                )["active_bytes.all.peak"] // 1000000
                # print('Memory end: {}'.format(mem_end))
                used_mem = mem_end - mem_start
            feature = ret_dict['outputs'][self.feature_axis]
            density = ret_dict['density']
        elif args.model == 'sparse_nlm':
            if args.task_is_adjacent and args.task_is_mnist_input:
                states_shape = states.size()
                states = states.view((-1,) + states_shape[2:])
                states = self.lenet(states)
                states = states.view(states_shape[:2] + (-1,))
                states = F.sigmoid(states)

            inp = [None, states, relations, ternaries][:args.nlm_breadth + 1]
            depth = None
            for i, vs in enumerate(inp):
                if torch.is_tensor(vs):
                    n = vs.shape[1]
                    dim = len(vs.shape) - 2
                    channel = vs.shape[-1]
                    inp[i] = [SparseHypergraph(v, n, dim, channel) for v in vs]
                    if args.nlm_sparsify_input:
                        for j in range(len(inp[i])):
                            inp[i][j].sparsify_by_max_value()
            if args.benchmark:
                empty_cache()
                torch.cuda.reset_peak_memory_stats()
                mem_start = torch.cuda.memory_stats(
                )["active_bytes.all.peak"] // 1000000
                # print('Memory start: {}'.format(mem_start))
                time_start = time.time()
                torch.cuda.synchronize()
            ret_dict = self.features(inp, depth=depth, plot_dir=args.plot_dir)
            if args.benchmark:
                torch.cuda.synchronize()
                run_time = time.time() - time_start
                mem_end = torch.cuda.memory_stats(
                )["active_bytes.all.peak"] // 1000000
                # print('Memory end: {}'.format(mem_end))
                used_mem = mem_end - mem_start

            f = ret_dict['outputs']
            for i, vs in enumerate(f):
                if vs is not None:
                    f[i] = torch.stack([v.to_dense() for v in vs], dim=0)
            feature = f[self.feature_axis]
            stat = ret_dict['stat']

        if not args.builtin_head:
            pred = self.pred(feature)
        else:
            pred = feature

        # print(feature.shape) torch.Size([4, 20, 20, 8])
        if args.plot_dir is not None:
            raise NotImplementedError()
        # print(pred.shape) torch.Size([4, 20, 20, 1])
        if not args.task_is_adjacent:
            pred = pred.squeeze(-1)
        # if args.task_is_connectivity or args.task_is_transitivity:
        #     pred = meshgrid_exclude_self(pred)  # exclude self-cycle
        monitors = dict()

        edge = feed_dict.edge

        batch_id = torch.arange(edge.size(0), device=edge.device)
        if edge.size(1) == 2:
            pred = pred[batch_id, edge[:, 0], edge[:, 1]]
        else:
            pred = pred[batch_id, edge[:, 0], edge[:, 1], edge[:, 2]]

        target = feed_dict.target.float()

        if args.ranking_loss:
            batch_size = (target > 0).sum().long().item()
            pos_pred = pred[target > 0].view(batch_size, 1)
            neg_pred = pred[target <= 0].view(batch_size, -1)
            diff = neg_pred - pos_pred + args.ranking_margin
            loss = torch.mean(torch.max(diff, torch.zeros_like(diff)))
            pred = torch.sigmoid(pred)
        elif args.contrastive_loss:
            batch_size = (target > 0).sum().long().item()
            pred = pred.view(batch_size, -1)
            log_softmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.mean(-log_softmax(pred)[:, 0])
            pred = torch.sigmoid(pred.flatten())
        else:
            loss = nn.BCELoss()(pred, target)

        monitors.update(task_loss=loss)

        monitors.update(binary_accuracy(
            target, pred, return_float=False, task=args.task))
        # ohem loss is unused.
        if args.ohem_size > 0:
            loss = loss.view(-1).topk(args.ohem_size)[0].mean()
        nlm_monitor = {}
        if 'sparsity' in ret_dict:
            nlm_monitor.update({f'sparsity {d+1}': sparsity for d,
                                sparsity in enumerate(ret_dict['sparsity'])})
        if 'stat' in ret_dict:
            for k, v in ret_dict['stat'].items():
                nlm_monitor_update(nlm_monitor, k, v)
        sparsity_loss = 0
        if args.model == 'nlm':
            nlm_monitor.update({f'{k} {d+1}': l.detach().cpu()
                                for k, v in density.items() for d, l in enumerate(v)})
            sparsity_loss = sum(density['hoyer']) / len(density['hoyer'])
        elif args.model == 'sparse_nlm':
            sparsity_loss, sparse_loss_max, l0, util, space_tot = 0, 1e-6, 0, 0, 1e-6
            nnz = 0
            for depth, l in enumerate(stat[args.nlm_sparse_loss]):
                for order, v in enumerate(l):
                    l0 += stat['l0'][depth][order] * \
                        stat['space_size'][depth][order]
                    util += stat['space_util'][depth][order] * \
                        stat['space_size'][depth][order]
                    nnz += stat['nnz'][depth][order]
                    space_tot += stat['space_size'][depth][order]
                    if depth != args.nlm_depth or order != self.feature_axis:
                        sparsity_loss += v * stat['multiplier'][depth][order]
                        sparse_loss_max += stat['space_size'][depth][order] * \
                            stat['multiplier'][depth][order]
            sparsity_loss /= sparse_loss_max
            l0 /= space_tot
            util /= space_tot
            nlm_monitor.update(sparsity_loss=sparsity_loss)
            nlm_monitor.update(l0=l0)
            nlm_monitor.update(space_util=util)
            nlm_monitor.update(nnz=nnz)
        nlm_monitor.update(run_time=run_time)
        nlm_monitor.update(used_mem=used_mem)
        monitors.update(nlm_monitor)
        if self.training:
            return loss + sparsity_loss * args.sparsity_loss_ratio, monitors, dict(pred=pred, target=target)
        else:
            for k, v in nlm_monitor.items():
                if torch.is_tensor(v):
                    nlm_monitor[k] = v.detach().cpu().item()
            result = dict(pred=pred, nlm_monitor=nlm_monitor, target=target)
            return result

# TODO: ADD A VALIDATION SET


def make_dataset(n, epoch_size, mode):
    if args.task_is_link_prediction:
        if mode == 'train':
            return SingleContrastiveLinkPredDataset(args.task, mode, num_neg_per_pos=args.train_num_neg_per_pos,
                                                    k=args.link_pred_k,
                                                    subgraph_size=args.subgraph_size,
                                                    directed=args.task_is_directed,
                                                    bridge=args.single_link_pred_bridge, node_label=args.node_label, shuffle=True)
        else:
            return SingleContrastiveLinkPredDataset(args.task, mode, num_neg_per_pos=args.test_num_neg_per_pos,
                                                    k=args.link_pred_k,
                                                    subgraph_size=args.test_subgraph_size,
                                                    directed=args.task_is_directed,
                                                    bridge=args.single_link_pred_bridge, node_label=args.node_label, shuffle=False)
    else:
        raise NotImplementedError


def dict_to_tb(dict, writer, epoch, mode):
    for key in dict:
        writer.add_scalar(key + '/' + mode, dict[key], epoch)
    writer.flush()


def detect_nan(named_parameters):
    ret = False
    for k, v in named_parameters:
        if torch.isnan(v).any():
            ret = True
            print(f'{k} has NaNs!')
            print(f'Value: {v}')
            print(f'grad: {v.grad}')
    return ret


def hit_at_k(pos_pred, neg_pred, k):
    # pos_pred: (bs, 1)
    # neg_pred: (bs, num_neg)
    diff = neg_pred - pos_pred
    greater = diff > 0
    return (torch.sum(greater, dim=1) < k).float().mean().item()


class MyTrainer(TrainerBase):
    def save_checkpoint(self, name):
        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))
            super().save_checkpoint(checkpoint_file)

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['epoch'] = self.current_epoch
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    data_iterator = {}
    epoch_sizes = {}

    def _prepare_dataset(self, epoch_size, mode):
        if mode == 'train':
            batch_size = args.batch_size * (1 + args.train_num_neg_per_pos)
            number = args.train_number
            subgraph = args.subgraph
            subgraph_size = args.subgraph_size
            start_num = args.start_num
            walk_length = args.walk_length
            do_calibrate = args.do_calibrate
        else:
            batch_size = args.test_batch_size * (1 + args.test_num_neg_per_pos)
            number = self.test_number
            subgraph = args.test_subgraph
            subgraph_size = args.test_subgraph_size
            start_num = args.test_start_num
            walk_length = args.test_walk_length
            do_calibrate = False

        dataset = make_dataset(number, epoch_size * batch_size, mode)
        # TODO: TEST EPOCH SIZE FIXED
        dataloader = JacDataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=min(epoch_size, args.num_workers))
        self.epoch_sizes[mode] = min(len(dataloader), epoch_size)
        self.data_iterator[mode] = dataloader.__iter__()

    def _get_data(self, index, meters, mode):
        feed_dict = self.data_iterator[mode].next()
        meters.update(number=feed_dict['n'].data.numpy().mean())
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)
        return feed_dict

    def _get_result(self, index, meters, mode):
        feed_dict = self._get_data(index, meters, mode)
        output_dict = self.model(feed_dict)
        pred, target = output_dict['pred'].flatten(
        ), output_dict['target'].flatten()
        batch_size = target.sum().long().item()
        pos_pred = pred[target > 0].view(batch_size, 1)
        neg_pred = pred[target <= 0].view(batch_size, -1)
        balanced_pred = torch.cat([pos_pred, neg_pred[:, :1]], dim=1)
        balanced_target = torch.cat(
            [torch.ones_like(pos_pred), torch.zeros_like(neg_pred[:, :1])], dim=1)
        result = binary_accuracy(
            balanced_target, balanced_pred, task=args.task)

        result['hit@1'] = hit_at_k(pos_pred, neg_pred, 1)
        result['hit@3'] = hit_at_k(pos_pred, neg_pred, 3)
        result['hit@10'] = hit_at_k(pos_pred, neg_pred, 10)
        succ = result['accuracy'] == 1.0
        meters.update(succ=succ)
        nlm_monitor = output_dict['nlm_monitor']
        meters.update(nlm_monitor)
        # wandb.log({f'{mode}_{self.test_number}_' +
        #    k: v for k, v in result.items()})
        # wandb.log({f'{mode}_{self.test_number}_' +
        #    k: v for k, v in nlm_monitor.items()})
        meters.update(result, n=target.size(0))
        message = '> {} iter={iter}, hit@10={hit@10:.4f}, \
balance_acc={balanced_accuracy:.4f}'.format(
            mode, iter=index, **meters.val)
        return message, dict(succ=succ, feed_dict=feed_dict)

    def _get_train_data(self, index, meters):
        return self._get_data(index, meters, mode='train')

    def _train_step(self, feed_dict, meters):
        ret = self.step(feed_dict)
        loss, monitors, output_dict, extras = ret
        meters.update(monitors)
        meters.update(loss=loss)
        # wandb.log(monitors)
        return 'Train: loss={loss:.4f}'.format(loss=loss), ret

    def _train_epoch(self, epoch_size):
        meters = super()._train_epoch(epoch_size)
        i = self.current_epoch
        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        if args.test_interval is not None and i % args.test_interval == 0:
            self.test()
        return meters

    def _early_stop(self, meters):
        return meters.avg['loss'] < args.early_stop_loss_thresh


def main(run_id):
    if args.dump_dir is not None:
        if args.runs > 1:
            args.current_dump_dir = os.path.join(args.dump_dir,
                                                 'run_{}'.format(run_id))
            io.mkdir(args.current_dump_dir)
        else:
            args.current_dump_dir = args.dump_dir

        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
        args.checkpoints_dir = os.path.join(
            args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)

    logger.info(format_args(args))

    model = Model()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    if args.use_gpu:
        model.cuda()
    # wandb.init('sparse-nlm', config=args)
    # wandb.watch(model, log_freq=1)
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)
    trainer = MyTrainer.from_args(model, optimizer, args)
    trainer.current_epoch = 0

    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)

    if args.test_only:
        return None, trainer.test()

    final_meters = trainer.train()
    trainer.save_checkpoint('last')

    return trainer.early_stopped, trainer.test()


if __name__ == '__main__':
    stats = []
    nr_graduated = 0

    for i in range(args.runs):
        graduated, test_meters = main(i)
        logger.info('run {}'.format(i + 1))

        if test_meters is not None:
            for j, meters in enumerate(test_meters):
                if len(stats) <= j:
                    stats.append(GroupMeters())
                stats[j].update(
                    number=meters.avg['number'], test_acc=meters.avg['accuracy'])

            for meters in stats:
                logger.info('number {}, test_acc {}'.format(meters.avg['number'],
                                                            meters.avg['test_acc']))

        if not args.test_only:
            nr_graduated += int(graduated)
            logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
            if graduated:
                for j, meters in enumerate(test_meters):
                    stats[j].update(grad_test_acc=meters.avg['accuracy'])
            if nr_graduated > 0:
                for meters in stats:
                    logger.info('number {}, grad_test_acc {}'.format(
                        meters.avg['number'], meters.avg['grad_test_acc']))
