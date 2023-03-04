import unittest
import os
import torch
import torch_xla.core.xla_model as xm
from transformers import BertForSequenceClassification, BertConfig
from fileutil import Platform
from dataset import get_dataloader, DatasetKind, PartitionKind, DataOption
import torch_xla.distributed.parallel_loader as pl


class TestCheckpoint(unittest.TestCase):

    def test_sst2_teacher(self):
        option = DataOption(train_batch_size=32,
                            eval_batch_size=100,
                            seed=42,
                            num_workers=4)
        loader = get_dataloader(DatasetKind.SST2,
                                PartitionKind.TEST,
                                option,
                                silence=True)
        device = xm.xla_device()
        cwd = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(cwd, './configs/bert_config.json')
        bert_config = BertConfig.from_json_file(cfg_path)
        bert_config.num_labels = 2
        teacher_model = BertForSequenceClassification(bert_config)
        state_dict = Platform.load_model(
            'gs://tianjin-distgen/sst2_teacher_model.pt',
            map_location=torch.device('cpu'))
        teacher_model.load_state_dict(state_dict)
        teacher_model.to(device=device)
        teacher_model.eval()

        correct = torch.tensor(0.0).to(device)
        total = torch.tensor(0.0).to(device)

        for idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = teacher_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct += torch.sum(torch.eq(predictions, batch["labels"]))
            total += torch.tensor(predictions.size(0))
            xm.mark_step()

        self.assertEqual(correct.item(), 807)
        self.assertEqual(total.item(), 872)


if __name__ == '__main__':
    unittest.main()
