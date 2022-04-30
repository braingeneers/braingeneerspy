import unittest
import braingeneers
import os
import distutils.core
import inspect


class TestSetup(unittest.TestCase):
    def test_setup_py(self):
        """ Simple init check on setup.py, this executes the code in setup.py """
        setup_py_path = os.path.split(os.path.dirname(inspect.getfile(braingeneers)))[0] + '/setup.py'
        distutils.core.run_setup(setup_py_path, stop_after='init')
        self.assertTrue(True)
