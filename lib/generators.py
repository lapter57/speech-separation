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

