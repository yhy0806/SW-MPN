import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from datasets.ESC50 import ESC50
from datasets.UrbanSound8K import urbansound8K
from .ESC48 import AudioSetDataset
from torch.utils.data import DataLoader, Dataset

dataset_dir = '/home/admin1/ARPL/dataset_48'

dataset_config = {
    'meta_csv': os.path.join(dataset_dir, "meta2.csv"),
    'audio_path': os.path.join(dataset_dir, "ESC_48")
}

#dataset_config = {
#    'meta_csv': os.path.join(dataset_dir, "UrbanSound8K", "metadata", "UrbanSound8K.csv"),
#    'audio_path': os.path.join(dataset_dir, "UrbanSound8K", "audio")
#}


def transform_function(x):
    return torch.tensor(x, dtype=torch.float32)


class AudioDatasetFilter(AudioSetDataset):
    """Audio Dataset with filtering based on known labels."""
    def __Filter__(self, known):
        audiopaths, waveforms, targets = self.audio_paths, np.array(self.waveform), np.array(self.targets)
        new_targets, mask = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                new_targets.append(known.index(targets[i]))
                mask.append(i)
        self.waveform, self.targets = np.squeeze(np.take(waveforms, mask, axis=0)), np.array(new_targets)
        self.audio_paths = np.array(self.audio_paths)[mask]


class AudioDatasetLoader(object):
    """DataLoader for Audio Dataset."""
    def __init__(self, meta_dir, audio_dir, known, batch_size=128, num_workers=8, use_gpu=True, transform=None):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 48))) - set(known))

        transform = transform_function

        pin_memory = True if use_gpu else False

        trainset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=True, transform=transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        testset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=False, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=False, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = DataLoader(outset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class urbansound8K_Filter(urbansound8K):
    """Audio Dataset with filtering based on known labels."""
    def __Filter__(self, known):
        audiopaths, waveforms, targets = self.audio_paths, np.array(self.waveform), np.array(self.targets)
        new_targets, mask = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                new_targets.append(known.index(targets[i]))
                mask.append(i)
        self.waveform, self.targets = np.squeeze(np.take(waveforms, mask, axis=0)), np.array(new_targets)
        self.audio_paths = np.array(self.audio_paths)[mask]


class urbansound8KLoader(object):
    """DataLoader for Audio Dataset."""
    def __init__(self, meta_dir, audio_dir, known, batch_size=128, num_workers=8, use_gpu=True, transform=None):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        transform = transform_function

        pin_memory = True if use_gpu else False

        trainset = urbansound8K_Filter(meta_dir, audio_dir, train=True, transform=transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        testset = urbansound8K_Filter(meta_dir, audio_dir, train=False, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = urbansound8K_Filter(meta_dir, audio_dir, train=False, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = DataLoader(outset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class ESC50Filter(ESC50):
    """Audio Dataset with filtering based on known labels."""
    def __Filter__(self, known):
        audiopaths, waveforms, targets = self.audio_paths, np.array(self.waveform), np.array(self.targets)
        new_targets, mask = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                new_targets.append(known.index(targets[i]))
                mask.append(i)
        self.waveform, self.targets = np.squeeze(np.take(waveforms, mask, axis=0)), np.array(new_targets)
        self.audio_paths = np.array(self.audio_paths)[mask]


class ESC50Loader(object):
    """DataLoader for Audio Dataset."""
    def __init__(self, meta_dir, audio_dir, known, batch_size=128, num_workers=8, use_gpu=True, transform=None):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 48))) - set(known))

        transform = transform_function

        pin_memory = True if use_gpu else False

        trainset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=True, transform=transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        testset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=False, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        outset = AudioDatasetFilter(meta_dir, audio_dir, fold=1, train=False, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = DataLoader(outset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--dataset', type=str, default='ESC_48',
                        help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | ESC_48")
    parser.add_argument('--audio_dir', type=str, default=dataset_config['audio_path'])
    parser.add_argument('--meta_dir', type=str, default=dataset_config['meta_csv'])
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    options = vars(args)
    from split import splits_2020 as splits

    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        unknown = list(set(list(range(0, 48))) - set(known))

        options.update(
            {
                'item': i,
                'known': known,
                'unknown': unknown,
            }
        )

        Data = AudioDatasetLoader(audio_dir=options['audio_dir'], meta_dir=options['meta_dir'], known=options['known'],
                                  batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        for f, (x, y) in enumerate(trainloader):
            pass