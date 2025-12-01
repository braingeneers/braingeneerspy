"""
Copyright (c) 2023 Braingeneers. All rights reserved.

braingeneers: braingeneerspy
"""

from __future__ import annotations
import warnings

from importlib.metadata import version as _pkg_version, PackageNotFoundError

from . import utils
from .utils.configure import (
    set_default_endpoint,
    get_default_endpoint,
    skip_unittest_if_offline,
)

try:  # preferred: read version from installed package metadata
    VERSION = _pkg_version("braingeneers")
except PackageNotFoundError:  # e.g. running from a source tree without an installed dist
    VERSION = "0.0.0.0000"

__version__ = VERSION

__all__ = ("set_default_endpoint", "get_default_endpoint", "skip_unittest_if_offline", "utils")

# Deprecated imports are allowed for backwards compatibility.
# This code should be removed in the future. This was added 27apr2022 by David Parks.
def __getattr__(name):
    if name == 'neuron':
        warnings.warn(
            message='braingeneers.neuron has been deprecated, please import braingeneers.analysis.neuron.',
            category=DeprecationWarning,
        )
        from braingeneers.analysis import neuron
        return neuron

    if name == 'datasets_electrophysiology':
        warnings.warn(
            message='braingeneers.datasets_electrophysiology has been deprecated, '
                    'please import braingeneers.data.datasets_electrophysiology.',
            category=DeprecationWarning,
        )
        from braingeneers.data import datasets_electrophysiology
        return datasets_electrophysiology

    if name == 'datasets_fluidics':
        warnings.warn(
            message='braingeneers.datasets_fluidics has been deprecated, '
                    'please import braingeneers.data.datasets_fluidics.',
            category=DeprecationWarning,
        )
        from braingeneers.data import datasets_fluidics
        return datasets_fluidics

    if name == 'datasets_imaging':
        warnings.warn(
            message='braingeneers.datasets_imaging has been deprecated, '
                    'please import braingeneers.data.datasets_imaging.',
            category=DeprecationWarning,
        )
        from braingeneers.data import datasets_imaging
        return datasets_imaging

    if name == 'datasets_neuron':
        warnings.warn(
            message='braingeneers.datasets_neuron has been deprecated, '
                    'please import braingeneers.data.datasets_neuron.',
            category=DeprecationWarning,
        )
        from braingeneers.data import datasets_neuron
        return datasets_neuron

    else:
        raise AttributeError(name)
