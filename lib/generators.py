import sys
sys.path.append("../lib")
import utils
import numpy as np
from tensorflow import keras

class AudioGenerator(keras.utils.Sequence):
    def __init__(self, mix_files, crm_files, n_speakers, batch_size=6, shuffle=True):
        self.Xdim = (298, 257, 2)
        self.Ydim = (298, 257, 2, n_speakers)
        self.batch_size = batch_size
        self.mix_files = mix_files
        self.crm_files = crm_files
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
            cRMs = utils.find_paths_contains(mix_filename, self.crm_files)
            for j, cRM in enumerate(cRMs):
                y[i, :, :, :, j] = np.load(cRM)
        return X, y
