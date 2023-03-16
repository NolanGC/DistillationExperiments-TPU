import unittest
from dataset import get_dataloader, DatasetKind, PartitionKind, DataOption

class TestDataset(unittest.TestCase):
    def test_len(self):
        option = DataOption(train_batch_size=1, eval_batch_size=1, seed=42, num_workers=4)

        train = get_dataloader(DatasetKind.SST2, PartitionKind.TRAIN, option, silence=True)
        self.assertEqual(len(train), 67349)

        test = get_dataloader(DatasetKind.SST2, PartitionKind.TEST, option, silence=True)
        self.assertEqual(len(test), 872)

    def test_subsample(self):
        option = DataOption(train_batch_size=1, eval_batch_size=1, seed=42, num_workers=4, subsample_fraction=0.1)
        train = get_dataloader(DatasetKind.SST2, PartitionKind.TRAIN, option, silence=True)
        self.assertEqual(len(train), 6735)

        test = get_dataloader(DatasetKind.SST2, PartitionKind.TEST, option, silence=True)
        self.assertEqual(len(test), 872)
        
if __name__ == '__main__':
    unittest.main()
