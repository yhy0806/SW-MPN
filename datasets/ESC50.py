import os
from functools import partial
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import librosa

dataset_dir = r'D:\pyproject\Openset\dataset_48\ESC-50'
dataset_config = {
    'meta_csv': os.path.join(dataset_dir, "meta", "esc50.csv"),
    'audio_path': os.path.join(dataset_dir, "audio"),
}


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


def pydub_augment(waveform, gain_augment=12):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


def get_roll_func(axis=1, shift=None, shift_range=4000):
    return partial(roll_func, axis=axis, shift=shift, shift_range=shift_range)


# roll waveform (over time)
def roll_func(b, axis=1, shift=None, shift_range=4000):
    x = b[0]
    others = b[1:]
    x = torch.as_tensor(x)
    sf = shift
    if shift is None:
        sf = int(np.random.random_integers(-shift_range, shift_range))
    return (x.roll(sf, axis), *others)


class PreprocessDataset(Dataset):
    """A base preprocessing dataset representing a preprocessing step of a Dataset preprocessed on the fly.
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor

    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class ESC50(Dataset):
    def __init__(self, meta_csv, audiopath, fold, train=False, resample_rate=32000, classes_num=48,
                 clip_length=5, gain_augment=12, transform=None):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.resample_rate = resample_rate
        self.meta_csv = meta_csv
        self.df = pd.read_csv(meta_csv, encoding='GB2312')
        if train:  # training all except this
            print(f"Dataset training fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold != fold]
            print(f" for training remains {len(self.df)}")
        else:
            print(f"Dataset testing fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold == fold]
            print(f" for testing remains {len(self.df)}")

        self.targets, self.waveform, self.audio_paths = [], [], []

        self.clip_length = clip_length * resample_rate
        self.classes_num = classes_num
        self.gain_augment = gain_augment
        self.audiopath = audiopath
        self.transform = transform

        # mel = AugmentMelSTFT(freqm=0, timem=0, fmin=0)

        for index in range(len(self.df['target'])):
            row = self.df.iloc[index]

            waveform, _ = librosa.load(os.path.join(self.audiopath, row.filename), sr=self.resample_rate, mono=True)
            waveform = torch.tensor(waveform)
            if self.gain_augment:
                waveform = pydub_augment(waveform, self.gain_augment)
            waveform = pad_or_truncate(waveform, self.clip_length)
            self.waveform.append(waveform)
            file_path = os.path.join(self.audiopath, row.filename)
            self.audio_paths.append(file_path)
            target = torch.tensor(row.target)
            self.targets.append(target)

        # data = _mel_forward(self.waveform, mel)
        # self.mel.append(data)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]
        filename = row.filename
        _, sr = librosa.load(os.path.join(self.audiopath, filename), sr=self.resample_rate, mono=True)
        wave, target = self.waveform[index], self.targets[index]

        if self.transform is not None:
            waveform = self.transform(wave)

        file_path = self.audio_paths[index]

        return file_path, waveform, target

    def __len__(self):
        return len(self.waveform)