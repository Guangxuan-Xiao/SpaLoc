#! /usr/bin/env python3
#

"""Implement Neural Logic Layers and Machines."""

import torch
import torch.nn as nn

from jacinle.logging import get_logger

from .modules.dimension import Expander, Reducer, Permutation
from .modules.neural_logic import LogicInference
from .modules._utils import print_groundings

__all__ = ['LogicLayer', 'LogicMachine']

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


class LogicLayer(nn.Module):
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
        reducer='minmax',
        verbose=False
    ):
        super().__init__()
        assert breadth > 0, 'Does not support breadth <= 0.'
        if breadth > 3:
            logger.warn(
                'Using LogicLayer with breadth > 3 may cause speed and memory issue.')

        self.max_order = breadth
        self.residual = residual
        self.reducer = reducer
        self.verbose = verbose
        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
        output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

        self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [
            nn.ModuleList() for _ in range(4)
        ]
        for i in range(self.max_order + 1):
            # collect current_dim from group i-1, i and i+1.
            current_dim = input_dims[i]
            if i > 0:
                expander = Expander(i - 1)
                self.dim_expanders.append(expander)
                current_dim += expander.get_output_dim(input_dims[i - 1])
            else:
                self.dim_expanders.append(None)

            if i + 1 < self.max_order + 1:
                if self.reducer == 'mean':
                    reducer = Reducer(i + 1, exclude_self, mean=True)
                else:
                    reducer = Reducer(i + 1, exclude_self, mean=False)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0:
                self.dim_perms.append(None)
                self.logic.append(None)
                output_dims[i] = 0
            else:
                perm = Permutation(i)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)
                self.logic.append(
                    LogicInference(current_dim, output_dims[i], logic_hidden_dim, norm))

        self.input_dims = input_dims
        self.output_dims = output_dims

        if self.residual:
            for i in range(len(input_dims)):
                self.output_dims[i] += input_dims[i]

    def forward(self, inputs):
        # print(self.input_dims)
        # for arity, grounding in enumerate(inputs):
        #     if torch.is_tensor(grounding):
        #         grounding = grounding.shape
        #     print(f'{arity}: {grounding}')
        assert len(inputs) == self.max_order + 1
        outputs = []
        for i in range(self.max_order + 1):
            # print(f'Collecting {i}')
            if self.verbose:
                print(f'\nCollecting Order {i}')
            # collect input f from group i-1, i and i+1.
            f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                # print(f'Get expand from {i - 1}')
                n = inputs[i].size(1) if i == 1 else None
                expanded = self.dim_expanders[i](inputs[i - 1], n)
                f.append(expanded)
                if self.verbose:
                    print(f'\nGet expanded predicates from order {i-1}:')
                    print_groundings(expanded)
            if i < len(inputs) and self.input_dims[i] > 0:
                # print(f'Get self {i}')
                f.append(inputs[i])
                if self.verbose:
                    print(f'\nGet same order predicates:')
                    print_groundings(inputs[i])
            if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                # print(f'Get reduce from {i + 1}')
                reduced = self.dim_reducers[i](inputs[i + 1])
                f.append(reduced)
                if self.verbose:
                    print(f'\nGet reduced predicates from order {i+1}:')
                    print_groundings(reduced)
            if len(f) == 0:
                # print(f'Output is none at {i}')
                output = None
            else:
                f = torch.cat(f, dim=-1)
                if self.verbose:
                    print('\nConcatenation:')
                    print_groundings(f)
                # print(f'Before perm: {f.shape}')
                f = self.dim_perms[i](f)
                if self.verbose:
                    print('\nPermutation:')
                    print_groundings(f)
                # print(f'After perm: {f.shape}')
                output = self.logic[i](f)
                if self.verbose:
                    print('\nLogic:')
                    print_groundings(output)
                # print(f'After logic: {output.shape}')
            if self.residual and self.input_dims[i] > 0:
                output = torch.cat([inputs[i], output], dim=-1)
            outputs.append(output)
        return outputs

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


class LogicMachine(nn.Module):
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
        reducer='minmax',
        verbose=False,
        builtin_head=False
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
        self.reducer = reducer
        self.verbose = verbose
        assert not (self.residual and self.io_residual), \
            'Only one type of residual connection is allowed at the same time.'

        # element-wise addition for vector
        def add_(x, y):
            for i in range(len(y)):
                x[i] += y[i]
            return x

        self.layers = nn.ModuleList()
        current_dims = input_dims
        total_output_dims = [0 for _ in range(self.breadth + 1)
                             ]  # for IO residual only
        for i in range(depth):
            # xgx
            # IO residual is unused.
            if i > 0 and io_residual:
                add_(current_dims, input_dims)
            # Not support output_dims as list or list[list] yet.
            if builtin_head and i == depth - 1:
                layer = LogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                                   exclude_self, False, 'sigmoid', clip, self.reducer, verbose=verbose)
            else:
                layer = LogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                                   exclude_self, residual, norm, clip, self.reducer, verbose=verbose)
            current_dims = layer.output_dims
            for mask in self.disable_mask:
                current_dims[mask] = 0
            current_dims = self._mask(current_dims, i, 0)
            if io_residual:
                add_(total_output_dims, current_dims)
            self.layers.append(layer)
            # print(f'{i}: {self.layers[i].input_dims}')
        # print(f'{depth}: {current_dims}')

        if io_residual:
            self.output_dims = total_output_dims
        else:
            self.output_dims = current_dims

    # Mask out the specific group-entry in layer i, specified by self.connections.
    # For debug usage.
    def _mask(self, a, i, masked_value):
        if self.connections is not None:
            assert i < len(self.connections)
            mask = self.connections[i]
            if mask is not None:
                assert len(mask) == len(a)
                a = [x if y else masked_value for x, y in zip(a, mask)]
        return a

    def forward(self, inputs, depth=None, plot_dir=None):
        outputs = [None for _ in range(self.breadth + 1)]
        f = inputs
        if self.verbose:
            for i, inp in enumerate(f):
                print(f'\nInput at Order {i}: ')
                print_groundings(inp)
        # depth: the actual depth used for inference
        if depth is None:
            depth = self.depth
        if not self.recursion:
            depth = min(depth, self.depth)

        def merge(x, y):
            if x is None:
                return y
            if y is None:
                return x
            return torch.cat([x, y], dim=-1)

        def hoyer_fn(v: torch.Tensor):
            # The smaller, the sparser
            N = v.numel()
            l1 = v.sum()
            l2 = torch.sqrt((v*v).sum())
            sqrt_N = N ** 0.5
            return (l1 / l2 - 1) / (sqrt_N - 1)

        def l0_fn(v: torch.Tensor, eps=0.1):
            return (v > eps).float().mean()

        def l1_fn(v: torch.Tensor):
            return v.mean()

        def calc_density(f):
            hoyer_tot = 0
            l0_tot = 0
            n = 0
            for arity, v in enumerate(f):
                # print(f'arity = {arity}')
                if torch.is_tensor(v):
                    n += 1
                    v = v.max(-1)[0]
                    l0 = l0_fn(v)
                    l0_tot += l0
                    hoyer = hoyer_fn(v)
                    hoyer_tot += hoyer
                    # print(
                    #     f'arity = {arity}, L0 = {l0}, Mean = {v.mean()}, Max: {v.max()}, Min: {v.min()}, Hoyer: {hoyer}')
            return hoyer_tot / n, l0_tot / n

        def clip(f, threshold=0):
            tot = 0
            cliped = 0
            for arity, v in enumerate(f):
                if torch.is_tensor(v):
                    v = v.max(-1)[0]
                    tot += v.numel()
                    mask = v < threshold
                    cliped += mask.sum()
                    f[arity][mask] = 0

        layer = None
        last_layer = None
        density = {'hoyer': [], 'l0': []}
        for i in range(depth):
            # xgx
            if self.verbose:
                print(f'\nIn layer {i + 1}:')
            if i > 0 and self.io_residual:
                for j, inp in enumerate(inputs):
                    f[j] = merge(f[j], inp)
            # To enable recursion, use scroll variables layer/last_layer
            # For weight sharing of period 2, i.e. 0,1,2,1,2,1,2,...
            if self.recursion and i >= 3:
                assert not self.residual
                layer, last_layer = last_layer, layer
            else:
                last_layer = layer
                layer = self.layers[i]
            f = layer(f)
            if self.verbose:
                for j, inp in enumerate(f):
                    print(f"\nOut layer {i + 1} at order {j}:")
                    print_groundings(inp)
            f = self._mask(f, i, None)
            if self.io_residual:
                for j, out in enumerate(f):
                    outputs[j] = merge(outputs[j], out)
            if self.clip > 0 and not self.training:
                clip(f, self.clip)
            hoyer, l0 = calc_density(f)
            density['hoyer'].append(hoyer)
            density['l0'].append(l0)
        if not self.io_residual:
            outputs = f
        ret_dict = {'outputs': outputs, 'density': density}
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
        'verbose',
        'reducer',
        'builtin_head',
    )

    __hyperparam_defaults__ = {
        'exclude_self': True,
        'io_residual': False,
        'residual': False,
        'recursion': False,
        'disable_mask': [],
        'norm': 'sigmoid',
        'clip': 0,
        'verbose': False,
        'reducer': 'minmax',
        'builtin_head': False,
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
        setattr(args, prefix + 'verbose', args.verbose)
        setattr(args, prefix + 'builtin_head', args.builtin_head)
        init_params = {k: getattr(args, prefix + k)
                       for k in cls.__hyperparams__}
        init_params.update(kwargs)

        return cls(**init_params)
