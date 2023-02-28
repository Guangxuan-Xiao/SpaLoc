#! /usr/bin/env python3
#

"""Command line interface, print or format args as string."""

from jacinle.utils.printing import kvformat
from jacinle.utils.printing import kvprint


def print_args(args):
    kvprint(args.__dict__)


def format_args(args):
    return kvformat(args.__dict__)
