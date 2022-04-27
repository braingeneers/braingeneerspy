import warnings


# Deprecated import braingeneers.utils.messaging is allowed for backwards compatibility.
# This code should be removed in the future. This was added 27apr2022 by David Parks.
def __getattr__(name):
    if name == 'messaging':
        warnings.warn(
            message='braingeneers.utils.messaging has been deprecated, please import braingeneers.iot.messaging.',
            category=DeprecationWarning,
        )
        from braingeneers.iot import messaging
        return messaging
    else:
        raise AttributeError(name)
