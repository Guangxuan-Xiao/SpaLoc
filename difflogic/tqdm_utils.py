#! /usr/bin/env python3
#

"""The utility functions for tqdm."""

from jacinle.utils.tqdm import tqdm_pbar

__all__ = ['tqdm_for']


def tqdm_for(total, func):
    """wrapper of the for function with message showing on the progress bar."""
    # Not support break cases for now.
    with tqdm_pbar(total=total) as pbar:
        for i in range(total):
            message = func(i)
            pbar.set_description(message)
            pbar.update()
