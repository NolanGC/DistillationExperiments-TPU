from enum import Enum
from datasets import load_dataset
from misc import silence as silence_context


class DatasetKind(Enum):
    SST2 = 1


class PartitionKind(Enum):
    TRAIN = 1
    TEST = 2


def get_dataset(dataset: DatasetKind,
                partition: PartitionKind,
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
            dataset.set_format(type='torch',
                               columns=[
                                   'input_ids', 'token_type_ids',
                                   'attention_mask', 'labels'
                               ])
        return dataset
    else:
        raise ValueError('invalid dataset kind %r' % dataset)
