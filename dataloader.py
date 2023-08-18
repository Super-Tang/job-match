import json

import torch
from torch.utils.data import Dataset


class InputExample:
    def __init__(self, task, text1, text2=None, label=None):
        self.task = task
        self.text1 = text1
        self.text2 = text2
        self.label = label


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataPreprocessor:
    def get_train_examples(self, path):
        df = json.load(open(path, 'r', encoding='utf-8'))
        return self._create_examples(df)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data):
        examples = []
        for row in data:
            task = row['job_description']
            text1 = row['item1'],
            text2 = row['item2']
            label = row['label']
            examples.append(InputExample(task, text1, text2, label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    if label_list is not None:
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

    features = []
    for index, example in enumerate(examples):
        encode_dict1 = tokenizer.encode_plus(example.task, example.text1,
                                             max_length=max_seq_length,
                                             truncation=True,
                                             add_special_tokens=True,
                                             pad_to_max_length=True,
                                             return_token_type_ids=True,
                                             return_attention_mask=True)

        encode_dict2 = tokenizer.encode_plus(example.task, example.text2,
                                             max_length=max_seq_length,
                                             truncation=True,
                                             add_special_tokens=True,
                                             pad_to_max_length=True,
                                             return_token_type_ids=True,
                                             return_attention_mask=True)

        input_ids = [encode_dict1['input_ids'], encode_dict2['input_ids']]
        input_mask = [encode_dict1['attention_mask'], encode_dict2['attention_mask']]
        segment_ids = [encode_dict1['token_type_ids'], encode_dict2['token_type_ids']]
        if label_list:
            label_id = label_map[example.label]
        else:
            label_id = 0
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
    return features


class MyDataset(Dataset):
    def __init__(self, features):
        self.input_ids0 = [torch.tensor(example.input_ids[0]).long() for example in features]
        self.input_mask0 = [torch.tensor(example.input_mask[0]).float() for example in features]
        self.segment_ids0 = [torch.tensor(example.segment_ids[0]).long() for example in features]
        self.input_ids1 = [torch.tensor(example.input_ids[1]).long() for example in features]
        self.input_mask1 = [torch.tensor(example.input_mask[1]).float() for example in features]
        self.segment_ids1 = [torch.tensor(example.segment_ids[1]).long() for example in features]
        self.label_id = [torch.tensor(example.label_id) for example in features]

    def __getitem__(self, index):
        data = {
            'input_ids0': self.input_ids0[index],
            'input_mask0': self.input_mask0[index],
            'segment_ids0': self.segment_ids0[index],
            'input_ids1': self.input_ids1[index],
            'input_mask1': self.input_mask1[index],
            'segment_ids1': self.segment_ids1[index],
            'label_id': self.label_id[index]
        }
        return data

    def __len__(self):
        return len(self.label_id)
