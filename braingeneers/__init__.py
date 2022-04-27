from braingeneers.configure import set_default_endpoint, get_default_endpoint, verify_optional_extras
import warnings


# Deprecated import braingeneers.neuron is allowed for backwards compatibility.
# This code should be removed in the future. This was added 27apr2022 by David Parks.
def __getattr__(name):
    if name == 'neuron':
        warnings.warn(
            message='braingeneers.neuron has been deprecated, please import braingeneers.analysis.neuron.',
            category=DeprecationWarning,
        )
        from braingeneers.analysis import neuron
        return neuron
    else:
        raise AttributeError(name)
