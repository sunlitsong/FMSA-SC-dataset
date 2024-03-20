import pickle

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data import DataLoader, Dataset


class MMDataset(Dataset):
    def __init__(self, args, mode='test'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'fgmsa': self.__init_fgmsa,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_fgmsa(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        self.text_cut = list(map(np.float32, data[self.mode]['text_cut']))
        self.args['feature_dims'][0] = self.text_cut[0].shape[2]
        self.audio_cut = list(map(np.float32, data[self.mode]['audio_cut']))
        self.args['feature_dims'][1] = self.audio_cut[0].shape[2]
        self.vision_cut = list(map(np.float32, data[self.mode]['vision_cut']))
        self.args['feature_dims'][2] = self.vision_cut[0].shape[2]
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        for m in "TAV":
            self.labels[m] = data[self.mode]['regression' + '_labels_' + m].astype(np.float32)

        for i in range(len(self.audio_cut)):
            self.audio_cut[i][self.audio_cut[i] == -np.inf] = 0

        if self.args.get('need_normalized'):
            for i in range(len(self.text_cut)):
                self.vision_cut[i] = np.transpose(self.vision_cut[i], (1, 0, 2))
                self.audio_cut[i] = np.transpose(self.audio_cut[i], (1, 0, 2))
                self.text_cut[i] = np.transpose(self.text_cut[i], (1, 0, 2))
                self.vision_cut[i] = np.mean(self.vision_cut[i], axis=0, keepdims=True)
                self.audio_cut[i] = np.mean(self.audio_cut[i], axis=0, keepdims=True)
                self.text_cut[i] = np.mean(self.text_cut[i], axis=0, keepdims=True)
                self.vision_cut[i][self.vision_cut[i] != self.vision_cut[i]] = 0
                self.audio_cut[i][self.audio_cut[i] != self.audio_cut[i]] = 0
                self.text_cut[i][self.text_cut[i] != self.text_cut[i]] = 0
                self.vision_cut[i] = np.transpose(self.vision_cut[i], (1, 0, 2)).squeeze()
                self.audio_cut[i] = np.transpose(self.audio_cut[i], (1, 0, 2)).squeeze()
                self.text_cut[i] = np.transpose(self.text_cut[i], (1, 0, 2)).squeeze()

    def __len__(self):
        return len(self.labels['M'])

    def __getitem__(self, index):
        sample = {
            'text_cut': torch.Tensor(self.text_cut[index]),
            'audio_cut': torch.Tensor(self.audio_cut[index]),
            'vision_cut': torch.Tensor(self.vision_cut[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }

        return sample


def __collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return __collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, dict):
        dic = {key: __collate_fn([d[key] for d in batch]) for key in elem
               if key not in ['text_cut', 'audio_cut', 'vision_cut']}
        dic.update({key: [d[key] for d in batch] for key in elem
                    if key in ['text_cut', 'audio_cut', 'vision_cut']})
        return dic
    elif isinstance(elem, list):
        return batch


def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True,
                       collate_fn=__collate_fn)
        for ds in datasets.keys()
    }

    return dataLoader
