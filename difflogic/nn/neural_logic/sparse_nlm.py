#! /usr/bin/env python3
#

"""Implement Neural Logic Layers and Machines."""

from os import access
import torch
import torch.nn as nn
from jacinle.logging import get_logger
from os.path import join
from .modules.sparse_dimension import SparseExpander, SparseReducer, SparsePermutation
from .modules.neural_logic import SparseLogicInference
import sparse_hypergraph as shg
from sparse_hypergraph import SparseHypergraph
from .modules._utils import plot
# torch.autograd.set_detect_anomaly(True)

__all__ = ['SparseLogicLayer', 'SparseLogicMachine']

logger = get_logger(__file__)


def _get_tuple_n(x, n, tp):
    """Get a length-n list of type tp."""
    assert tp is not list
    if isinstance(x, tp):
        x = [x, ] * n
    assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(
        tp)
    for i in x:
        assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
    return x


class SparseLogicLayer(nn.Module):
    """Logic Layers do one-step differentiable logic deduction.

    The predicates grouped by their number of variables. The inter group deduction
    is done by expansion/reduction, the intra group deduction is done by logic
    model.

    Args:
      breadth: The breadth of the logic layer.
      input_dims: the number of input channels of each input group, should consist
                  with the inputs. use dims=0 and input=None to indicate no input
                  of that group.
      output_dims: the number of output channels of each group, could
                   use a single value.
      logic_hidden_dim: The hidden dim of the logic model.
      exclude_self: Not allow multiple occurrence of same variable when
                    being True.
      residual: Use residual connections when being True.
    """

    def __init__(
        self,
        breadth,
        input_dims,
        output_dims,
        logic_hidden_dim,
        exclude_self=True,
        residual=False,
        norm='sigmoid',
        clip=0,
        verbose=False,
        importance='nn'
    ):
        super().__init__()
        assert breadth > 0, 'Does not support breadth <= 0.'
        if breadth > 3:
            logger.warn(
                'Using LogicLayer with breadth > 3 may cause speed and memory issue.')

        self.max_order = breadth
        self.residual = residual
        self.verbose = verbose
        self.importance = importance
        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
        output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

        self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [
            nn.ModuleList() for _ in range(4)
        ]
        if self.importance == 'nn':
            self.alpha = nn.ModuleList()

        for i in range(self.max_order + 1):
            # collect current_dim from group i-1, i and i+1.
            current_dim = input_dims[i]
            if i > 0:
                expander = SparseExpander(i - 1)
                self.dim_expanders.append(expander)
                current_dim += expander.get_output_dim(input_dims[i - 1])
            else:
                self.dim_expanders.append(None)

            if i + 1 < self.max_order + 1:
                reducer = SparseReducer(
                    i + 1, exclude_self)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0:
                self.dim_perms.append(None)
                self.logic.append(None)
                if self.importance == 'nn':
                    self.alpha.append(None)
                output_dims[i] = 0
            else:
                perm = SparsePermutation(i)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)
                self.logic.append(
                    SparseLogicInference(current_dim, output_dims[i], logic_hidden_dim, norm))
                if self.importance == 'nn':
                    self.alpha.append(SparseLogicInference(
                        current_dim, 1, logic_hidden_dim, 'sigmoid'))

        self.input_dims = input_dims
        self.output_dims = output_dims
        if self.residual:
            for i in range(len(input_dims)):
                self.output_dims[i] += input_dims[i]

    def forward(self, inputs):
        assert len(inputs) == self.max_order + 1
        outputs, alphas = [], []
        for i in range(self.max_order + 1):
            if self.verbose:
                print(f'\nOrder {i}')
            f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                expanded = self.dim_expanders[i](inputs[i - 1])
                if self.verbose:
                    print(f'\nGet expanded predicates from order {i-1}:')
                    print(expanded[0])
                f.append(expanded)
            if i < len(inputs) and self.input_dims[i] > 0:
                if self.verbose:
                    print('\nGet same order predicates:')
                    print(inputs[i][0])
                f.append(inputs[i])
            if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                reduced = self.dim_reducers[i](inputs[i + 1])
                if self.verbose:
                    print(f'\nGet reduced predicates from order {i+1}:')
                    print(reduced[0])
                # print('reduced[0].channel = ', reduced[0].channel)
                f.append(reduced)
            if len(f) == 0:
                output = None
                alpha = None
            else:
                bs = len(f[0])
                f = [shg.cat([t[j] for t in f]) for j in range(bs)]
                if self.verbose:
                    print('\nConcatenation:')
                    print(f[0])
                f = self.dim_perms[i](f)
                if self.verbose:
                    print('\nPermutation:')
                    print(f[0])
                output = self.logic[i](f)
                if self.importance == 'nn':
                    alpha = self.alpha[i](f)
                    for j in range(bs):
                        output[j].val = torch.mul(output[j].val, alpha[j].val)
                elif self.importance == 'max':
                    alpha = [t.reduce_channel('max') for t in output]
                elif self.importance == 'absmax':
                    alpha = [t.reduce_channel('absmax') for t in output]
                if self.verbose:
                    print('\nLogic')
                    print(output[0])
            if self.residual and self.input_dims[i] > 0:
                for idx, t in enumerate(output):
                    output[idx] = shg.cat([t, inputs[i][idx]])
            outputs.append(output)
            alphas.append(alpha)
        return outputs, alphas

    __hyperparams__ = (
        'breadth',
        'input_dims',
        'output_dims',
        'logic_hidden_dim',
        'exclude_self',
        'residual',
    )

    __hyperparam_defaults__ = {
        'exclude_self': True,
        'residual': False,
    }

    @classmethod
    def make_nlm_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(
            prefix + 'breadth',
            type='int',
            default=defaults['breadth'],
            metavar='N',
            help='breadth of the logic layer')
        parser.add_argument(
            prefix + 'logic-hidden-dim',
            type=int,
            nargs='+',
            default=defaults['logic_hidden_dim'],
            metavar='N',
            help='hidden dim of the logic model')
        parser.add_argument(
            prefix + 'exclude-self',
            type='bool',
            default=defaults['exclude_self'],
            metavar='B',
            help='not allow multiple occurrence of same variable')
        parser.add_argument(
            prefix + 'residual',
            type='bool',
            default=defaults['residual'],
            metavar='B',
            help='use residual connections')

    @classmethod
    def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ''
        else:
            prefix = str(prefix) + '_'

        setattr(args, prefix + 'input_dims', input_dims)
        setattr(args, prefix + 'output_dims', output_dims)
        init_params = {k: getattr(args, prefix + k)
                       for k in cls.__hyperparams__}
        init_params.update(kwargs)

        return cls(**init_params)


class SparseLogicMachine(nn.Module):
    """Neural Logic Machine consists of multiple logic layers."""

    def __init__(
        self,
        depth,
        breadth,
        input_dims,
        output_dims,
        logic_hidden_dim,
        exclude_self=True,
        residual=False,
        io_residual=False,
        recursion=False,
        connections=None,
        disable_mask=[],
        norm='sigmoid',
        clip=0,
        builtin_head=False,
        verbose=False,
        importance='nn',
        sparsify_input=False,
        sparsify_inter=False
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.recursion = recursion
        self.connections = connections
        self.disable_mask = disable_mask
        self.clip = clip
        self.sparsify = True
        self.builtin_head = builtin_head
        self.verbose = verbose
        self.sparsify_input = sparsify_input
        self.sparsify_inter = sparsify_inter
        # element-wise addition for vector

        def add_(x, y):
            for i in range(len(y)):
                x[i] += y[i]
            return x

        self.layers = nn.ModuleList()
        current_dims = input_dims

        for i in range(depth):
            # Not support output_dims as list or list[list] yet.
            if builtin_head and i == depth - 1:
                layer = SparseLogicLayer(breadth, current_dims, 1, logic_hidden_dim,
                                         exclude_self, False, 'sigmoid', clip, verbose, importance)
            else:
                layer = SparseLogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                                         exclude_self, residual, norm, clip, verbose, importance)
            current_dims = layer.output_dims
            for mask in self.disable_mask:
                current_dims[mask] = 0
            current_dims = self._mask(current_dims, i, 0)
            self.layers.append(layer)
        self.output_dims = current_dims

    def _mask(self, a, i, masked_value):
        if self.connections is not None:
            assert i < len(self.connections)
            mask = self.connections[i]
            if mask is not None:
                assert len(mask) == len(a)
                a = [x if y else masked_value for x, y in zip(a, mask)]
        return a

    def forward(self, inputs, depth=None, plot_dir=None):
        f = inputs
        if plot_dir is not None:
            plot(f, join(plot_dir, 'input'))
        depth = self.depth
        stat = {
            'hoyer': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'hoyer_square': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'multiplier': [[1] * (self.breadth + 1) for _ in range(self.depth)],
            'l1': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'l2': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'l0': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'nnz': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'space_util': [[0] * (self.breadth + 1) for _ in range(self.depth)],
            'space_size': [[0] * (self.breadth + 1) for _ in range(self.depth)]
        }

        for d in range(depth):
            if self.verbose:
                print(f'\nIn layer {d + 1}:')
            f, alpha = self.layers[d](f)
            if plot_dir is not None:
                plot(f, join(plot_dir, f'l{d}'))
            for order, ts in enumerate(alpha):
                if ts is None:
                    continue
                bs = len(ts) + 1e-6

                for idx, t in enumerate(ts):
                    if self.verbose:
                        print(f"\nOrder {order} before sparsify:")
                        print(f[order][0])
                    if not self.training and self.sparsify_inter:
                        f[order][idx].sparsify_by_importance(t)
                        t.sparsify_by_max_value()
                    n = t.n
                    stat['hoyer'][d][order] += t.hoyer() / bs
                    stat['hoyer_square'][d][order] += t.hoyer_square() / bs
                    stat['l1'][d][order] += t.l1() / bs
                    stat['l2'][d][order] += t.l2() / bs
                    stat['l0'][d][order] += t.l0() / bs
                    stat['nnz'][d][order] += t.num_nonzero_entries() / bs
                    stat['space_util'][d][order] += t.sparsity() / bs
                    stat['space_size'][d][order] += t.space_size / bs

                if d == depth - 1 or order == self.breadth:
                    stat['multiplier'][d][order] = 1
                else:
                    stat['multiplier'][d][order] = n

                if self.verbose:
                    print(f"\nOrder {order} after sparsify:")
                    print(f[order][0])
                    print(f"\nImportance value {order} after sparsify:")
                    print(alpha[order][0])

        ret_dict = {'outputs': f, 'stat': stat}
        if self.verbose:
            print(f'Statistics: {stat}')
        return ret_dict

    __hyperparams__ = (
        'depth',
        'breadth',
        'input_dims',
        'output_dims',
        'logic_hidden_dim',
        'exclude_self',
        'io_residual',
        'residual',
        'recursion',
        'disable_mask',
        'norm',
        'clip',
        'builtin_head',
        'verbose',
        'importance',
        'sparsify_input',
        'sparsify_inter'
    )

    __hyperparam_defaults__ = {
        'exclude_self': True,
        'io_residual': False,
        'residual': False,
        'recursion': False,
        'disable_mask': [],
        'norm': 'sigmoid',
        'clip': 0,
        'builtin_head': False,
        'verbose': False,
        'importance': 'nn',
        'sparsify_input': False,
        'sparsify_inter': False
    }

    def sparsify_on(self):
        self.sparsify = True

    @classmethod
    def make_nlm_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(
            prefix + 'depth',
            type=int,
            default=defaults['depth'],
            metavar='N',
            help='depth of the logic machine')
        parser.add_argument(
            prefix + 'breadth',
            type=int,
            default=defaults['breadth'],
            metavar='N',
            help='breadth of the logic machine')
        parser.add_argument(
            prefix + 'logic-hidden-dim',
            type=int,
            nargs='+',
            default=defaults['logic_hidden_dim'],
            metavar='N',
            help='hidden dim of the logic model')
        parser.add_argument(
            prefix + 'exclude-self',
            type='bool',
            default=defaults['exclude_self'],
            metavar='B',
            help='not allow multiple occurrence of same variable')
        parser.add_argument(
            prefix + 'io-residual',
            type='bool',
            default=defaults['io_residual'],
            metavar='B',
            help='use input/output-only residual connections')
        parser.add_argument(
            prefix + 'residual',
            type='bool',
            default=defaults['residual'],
            metavar='B',
            help='use residual connections')
        parser.add_argument(
            prefix + 'recursion',
            type='bool',
            default=defaults['recursion'],
            metavar='B',
            help='use recursion weight sharing')
        parser.add_argument(
            prefix + 'norm',
            type=str,
            default='sigmoid',
            help='norm function to get probability'
        )
        parser.add_argument(
            prefix+'reducer',
            type=str,
            default='minmax',
            help='method to value edge importance'
        )
        parser.add_argument(
            prefix+'disable-mask',
            type=str,
            default='',
            help='disable some predicates in nlm'
        )
        parser.add_argument(
            prefix+'importance',
            type=str,
            default='nn',
            help='method to value edge importance'
        )
        parser.add_argument(
            prefix+'clip',
            type=float,
            default=0,
            help='clip intermediate predicates'
        )
        parser.add_argument(
            prefix+'sparsify-input',
            action='store_true',
            help='sparsify input'
        )
        parser.add_argument(
            prefix+'sparsify-inter',
            action='store_true',
            help='sparsify intermediate embeddings'
        )

    @classmethod
    def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ''
        else:
            prefix = str(prefix) + '_'
        setattr(args, prefix + 'input_dims', input_dims)
        setattr(args, prefix + 'output_dims', output_dims)
        setattr(args, prefix + 'builtin_head', args.builtin_head)
        setattr(args, prefix + 'verbose', args.verbose)
        init_params = {k: getattr(args, prefix + k)
                       for k in cls.__hyperparams__}
        init_params.update(kwargs)

        return cls(**init_params)
