from enum import Enum
from datasets import load_dataset
from misc import silence as silence_context
import torch
import torch_xla.core.xla_model as xm
from pydantic.dataclasses import dataclass
from fileutil import Platform
import torch.nn.functional as F

@dataclass
class DataOption:
    train_batch_size: int = 4
    eval_batch_size: int = 100
    seed: int = 42
    num_workers: int = 4


class DatasetKind(Enum):
    SST2 = 1


class PartitionKind(Enum):
    TRAIN = 1
    TEST = 2


def get_dataloader(dataset: DatasetKind,
                   partition: PartitionKind,
                   option: DataOption,
                   silence: bool = False):
    if dataset == DatasetKind.SST2:
        from transformers import BertTokenizer
        split_map = {
            PartitionKind.TRAIN: "train",
            PartitionKind.TEST: "validation",
        }

        with silence_context(enable=silence):
            split = split_map[partition]
            dataset = load_dataset('glue', 'sst2', split=split)

            def rename(examples):
                return {"labels": examples["label"]}

            dataset = dataset.map(rename, batched=True)
            dataset = dataset.remove_columns(['label'])

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            MAX_LENGTH = 128

            def tokenize(e):
                return tokenizer(e['sentence'],
                                 truncation=True,
                                 padding='max_length',
                                 max_length=MAX_LENGTH)

            dataset = dataset.map(tokenize, batched=True)
        
        if partition == PartitionKind.TRAIN:
            all_el2n = []
            # Load EL2N score.
            for i in range(10):
                path = f"gs://tianjin-distgen/sst2/el2n-raw-{i}.pt"
                content = Platform.load_model(path)
                pred = torch.tensor(content["pred"])
                labels = torch.tensor(content["labels"])
                
                one_hot_labels = F.one_hot(labels, num_classes=2)
                prob_pred = F.softmax(pred, dim=-1)

                el2n = torch.norm(prob_pred - one_hot_labels, 2, dim=-1)
                all_el2n.append(el2n)

            all_el2n = torch.vstack(all_el2n)
            avg_el2n = torch.mean(all_el2n, dim=0).cpu().numpy()

            def assign_el2n(examples, idx):
                return {"el2n": avg_el2n[idx]}
            dataset = dataset.map(assign_el2n, batched=True, 
                                  with_indices=True)
            dataset.set_format(type='torch',
                                columns=[
                                    'input_ids', 'token_type_ids',
                                    'attention_mask', 'labels', 'el2n'
                                ])
            assert torch.all(dataset["labels"] == labels).item()
        else:
            dataset.set_format(type='torch',
                                columns=[
                                    'input_ids', 'token_type_ids',
                                    'attention_mask', 'labels'
                                ])

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True if partition == PartitionKind.TRAIN else False,
            seed=option.seed)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=option.train_batch_size if partition == PartitionKind.TRAIN else option.eval_batch_size,
            sampler=sampler,
            num_workers=option.num_workers,
            drop_last=False)

        return loader
    else:
        raise ValueError('invalid dataset kind %r' % dataset)
