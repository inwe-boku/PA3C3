# -*- coding: utf-8 -*-

"""Top-level package for spatialnc."""

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'unknown'
