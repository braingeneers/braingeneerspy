import braingeneers
import smart_open


braingeneers.set_default_endpoint()


# noinspection PyProtectedMember
def open(*args, **kwargs):
    """
    Simple hand off to smart_open, this explicit handoff is required because open can reference a
    new function if configure.set_default_endpoint is called in the future.
    """
    return braingeneers.utils.configure._open(*args, **kwargs)
