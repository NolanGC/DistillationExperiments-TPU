import unittest
from dataset import get_dataset, DatasetKind, PartitionKind

class TestDataset(unittest.TestCase):
    def test_len(self):
        train = get_dataset(DatasetKind.SST2, PartitionKind.TRAIN, silence=True)
        self.assertEqual(len(train), 67349)

        test = get_dataset(DatasetKind.SST2, PartitionKind.TEST, silence=True)
        self.assertEqual(len(test), 872)

if __name__ == '__main__':
    unittest.main()
