"""ssa_rs: Rust-accelerated drop-in replacement for py-ssa-lib (SSA / MSSA).

    from ssa_rs.ssa_collection import SSA, MSSA          # was py_ssa_lib.ssa_collection
    from ssa_rs.svd_strategies import FullSVD, RandomizedSVD, EigenSVD
    from ssa_rs.weights_strategies import PrecomputedWeights, ...
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import LinAlgError
from .ssa_collection import BaseSSA, SSA, MSSA
from .svd_strategies import SVDStrategy, FullSVD, RandomizedSVD, EigenSVD, LanczosSVD, ToeplitzSVD
from .weights_strategies import (
    WeightingStrategy,
    UniformWeights,
    PrecomputedWeights,
    DistanceWeights,
    ClusterAwareWeights,
)
from .stat_tools import bootstrap_prediction_intervals
from . import rssa

try:
    __version__ = version("ssa-rs")
except PackageNotFoundError:
    __version__ = "Package not found"
