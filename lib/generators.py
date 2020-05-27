import os
import sys
sys.path.append("../lib")
import utils
import numpy as np
from tensorflow import keras
import torch
from audio import Audio
from torch.utils.data import Dataset, DataLoader

def create_dataloader(config, train):

    def ao_collate_fn(batch):
        target_list = list()
        mix_list = list()

        for mix, targets in batch:
            mix_list.append(mix)
            target_list.append(targets)

        target_list = torch.stack(target_list, dim=0)
        mix_list = torch.stack(mix_list, dim=0)

        return mix_list, target_list

    def av_collate_fn(batch):
        emb_list = list()
        target_list = list()
        mix_list = list()

        for mix, embs, targets in batch:
            mix_list.append(mix)
            target_list.append(targets)
            emb_list.append(embs)

        emb_list = torch.stack(emb_list, dim=0)
        target_list = torch.stack(target_list, dim=0)
        mix_list = torch.stack(mix_list, dim=0)

        return mix_list, emb_list, target_list

    dataset = AudioDataset(config, True) if config.train.model == 'ao' else AudioVisualDataset(config, True)
    collate_fn = ao_collate_fn if config.train.model == 'ao' else av_collate_fn

    if train:
        return DataLoader(dataset=dataset,
                          batch_size = config.train.batch_size,
                          shuffle=True,
                          num_workers = config.train.num_workers,
                          collate_fn=ao_collate_fn if config.train.model == 'ao' else av_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

    return DataLoader(dataset=AudioDataset(config, False) if config.train.model == 'ao' else AudioVisualDataset(config, False),
                      collate_fn=collate_fn,
                      batch_size=1, shuffle=False, num_workers=0)

class CustomDataset(Dataset):
    def __init__(self, config, train):
        self.config = config
        self.data_dir = os.path.join(config.data.audio.path, "train") if train else os.path.join(config.data.audio.path, "test")
        self.audio_handler = Audio(config)

        self.target_list = utils.get_files(os.path.join(self.data_dir, "clean"))
        self.mix_list = np.random.shuffle(utils.get_files(os.path.join(self.data_dir, "mix")))

    def __len__(self):
        return len(self.mix_list)

    def get_data(self, idx):
        mix_file = self.mix_list[idx]
        mix = torch.Tensor(np.load(mix_file))
        mix_filename = utils.basename(mix_file)
        return mix, mix_filename.split(".")


class AudioDataset(CustomDataset):
    def __init__(self, config, train):
        super().__init__(config, train)

    def __getitem__(self, idx):
        mix, target_filenames = self.get_data(idx)
        targets = list()
        for filename in target_filenames:
            targets.append(utils.find_paths_contains(filename, self.target_list)[0])
        targets = list(map(lambda t: torch.Tensor(np.load(t)), targets))
        return mix, torch.stack(targets, dim=3), None

class AudioVisualDataset(CustomDataset):
    def __init__(self, config, train):
        super().__init__(config, train)
        self.emb_list = utils.get_files(os.path.join(config.data.video.emb_path, self.data_dir))

    def __getitem__(self, idx):
        mix, target_filenames = self.get_data(idx)
        targets = list()
        embs = list()
        for filename in target_filenames:
            targets.append(utils.find_paths_contains(filename, self.target_list)[0])
            embs.append(utils.find_paths_contains(filename, self.emb_list)[0])
        targets = list(map(lambda t: torch.Tensor(np.load(t)), targets))
        embs = list(map(lambda e: torch.Tensor(np.load(e)), embs))
        return mix, torch.stack(targets, dim=3), embs.stack(embs, dim=3)


class AudioGenerator(keras.utils.Sequence):
    def __init__(self, config, mix_files, clean_files, shuffle=True):
        self.Xdim = (config.audio.num_freq, config.audio.num_time, 2)
        self.Ydim = (config.audio.num_freq, config.audio.num_time, 2, config.data.num_speakers)
        self.batch_size = config.train.batch_size
        self.mix_files = mix_files
        self.clean_files = clean_files
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.mix_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        mix_temp = [self.mix_files[k] for k in indexes]
        X, y = self.__data_generation(mix_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.mix_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, mix_temp):
        X = np.empty((self.batch_size, *self.Xdim))
        y = np.empty((self.batch_size, *self.Ydim))
        for i, ID in enumerate(mix_temp):
            X[i,] = np.load(ID)
            mix_filename = utils.basename(ID)
            clean_filenames = mix_filename.split(".")
            cleans = np.array([])
            for cf in clean_filenames:
                cleans = np.append(cleans, utils.find_paths_contains(cf, self.clean_files))
            for j, clean in enumerate(cleans):
                y[i, :, :, :, j] = np.load(clean)
        return X, y

class AVGenerator(keras.utils.Sequence):
    
    def __init__(self, mix_files, crm_files, emb_files,
                 n_speakers, batch_size=6, shuffle=True):
        self.X1dim = (298, 257, 2)
        self.X2dim = (75, 1, 512, n_speakers)
        self.Ydim = (298, 257, 2, n_speakers)
        self.batch_size = batch_size
        self.mix_files = mix_files
        self.crm_files = crm_files
        self.emb_files = emb_files
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.mix_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        mix_temp = [self.mix_files[k] for k in indexes]
        [X1, X2], y = self.__data_generation(mix_temp)
        return [X1, X2], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.mix_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, mix_temp):
        X1 = np.empty((self.batch_size, *self.X1dim))
        X2 = np.empty((self.batch_size, *self.X2dim))
        y = np.empty((self.batch_size, *self.Ydim))

        for i, ID in enumerate(mix_temp):
            X1[i,] = np.load(ID)
            mix_filename = utils.basename(ID)

            embs = np.empty(0)
            emb_filenames = mix_filename.split(".")
            for emb_filename in emb_filenames:
                embs = np.append(embs, utils.find_paths_contains(emb_filename, self.emb_files))
            for j, emb in enumerate(embs):
                X2[i, :, :, :, j] = np.load(emb)

            cRMs = utils.find_paths_contains(mix_filename, self.crm_files)
            for j, cRM in enumerate(cRMs):
                y[i, :, :, :, j] = np.load(cRM)

        return [X1, X2], y 
