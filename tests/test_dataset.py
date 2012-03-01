import unittest
from blockplayer import dataset

class Test(unittest.TestCase):
    @unittest.skip
    def test_download(self):
        dataset.download()

    @unittest.skip
    def test_dataset(self):
        dataset.load_random_dataset()


if __name__ == '__main__':
    unittest.main()
