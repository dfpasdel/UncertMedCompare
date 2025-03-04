import unittest

loader = unittest.TestLoader()
start_dir = './UncertMedCompare'
suite = loader.discover(start_dir, pattern="*.py")

runner = unittest.TextTestRunner()
runner.run(suite)