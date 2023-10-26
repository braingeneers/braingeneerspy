import os
import pathlib
import platform
import warnings


# At import time add the Maxwell custom H5 compression plugin environment variable if it's not already set.
# This is necessary to enable all V2 maxwell H5 datafiles to be readable.
# We have included the compiled plugin binaries in the braingeneerspy source under braingeneerspy/data/mxw_h5_plugin/*.
if 'HDF5_PLUGIN_PATH' not in os.environ:
    system = platform.system()
    machine = platform.machine()

    plugin_arch_dir = \
        'Linux' if system == 'Linux' else \
        'Windows ' if system == 'Windows' else \
        'Mac_arm64' if system == 'Darwin' and machine == 'arm64' else \
        'Mac_x86_64' if system == 'Darwin' and machine == 'x86_64' else \
        None

    if plugin_arch_dir is None:
        warnings.warn(f'System [{system}] and machine [{machine}] architecture is not supported '
                      f'by the Maxwell HDF5 compression plugin. The Maxwell data reader will not '
                      f'work for V2 HDF5 files on this system.')
    else:
        os.environ['HDF5_PLUGIN_PATH'] = os.path.join(
            pathlib.Path(__file__).parent.resolve(),  # path to this __init__.py file
            'mxw_h5_plugin',  # sudirectory where maxwell plugins are stored (for all architectures)
            plugin_arch_dir  # architecture specific sudirectory where the system-specific plugin is stored
        )
