from enum import Enum
from datasets import load_dataset
from misc import silence as silence_context
import torch
import torch_xla.core.xla_model as xm
from pydantic.dataclasses import dataclass
from fileutil import Platform
import torch.nn.functional as F
import numpy as np 

@dataclass
class DataOption:
    train_batch_size: int = 4
    eval_batch_size: int = 100
    seed: int = 42
    num_workers: int = 4
    el2n_threshold : float = None

    # By default, difficult (high-EL2N) examples use teacher outputs during training, easy ones use onehot labels.
    # By setting this flag to true the reverse is true -- difficult examples uses onehot labels.
    el2n_invert_filter : bool = False
    subsample_fraction : float = None

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
            avg_el2n = torch.mean(all_el2n, dim=0)

            if option.el2n_threshold is not None:
                soft_label_mask = avg_el2n > option.el2n_threshold
                if option.el2n_invert_filter:
                    soft_label_mask = torch.logical_not(soft_label_mask)
            else:
                soft_label_mask = torch.ones_like(avg_el2n).bool()
            soft_label_mask = soft_label_mask.cpu().numpy()

            def assign_soft_label_mask(examples, idx):
                return {"soft_label_mask": soft_label_mask[idx]}

            dataset = dataset.map(assign_soft_label_mask, batched=True, 
                                  with_indices=True)
            dataset.set_format(type='torch',
                                columns=[
                                    'input_ids', 'token_type_ids',
                                    'attention_mask', 'labels', 
                                    'soft_label_mask'
                                ])
            assert torch.all(dataset["labels"] == labels).item()

            if option.subsample_fraction is not None:
                subsample_indices = np.arange(len(dataset))
                np.random.RandomState(option.seed).shuffle(subsample_indices)
                subsample_indices = subsample_indices[:int(len(dataset) * option.subsample_fraction)]
                print(subsample_indices)
                dataset = torch.utils.data.Subset(dataset, subsample_indices.tolist())

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
