import numpy as np


class OneDimensionalData():
    """One Dimensional dataset."""

    def __init__(self, InInterval, num, batch_size):
        """
        Args:
            InInterval (list):
            num (int):
        """
        self.batch_size = batch_size

        self.data = np.linspace(InInterval[0], InInterval[1], num, dtype=np.float32)
        np.random.shuffle(self.data)
        self.batches = np.reshape(self.data, (int(len(self.data)/self.batch_size), self.batch_size, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def GetData(self):
        return self.data

    def GetBatches(self):
        return self.batches


class TwoDimensionalData():
    """Two Dimensional dataset"""

    def __init__(self, InInterval, num, batch_size):
        self.data = np.random.uniform(low=InInterval[0], high=InInterval[-1], size=(num, 3))
        self.batches = np.reshape(self.data, (int(num/batch_size), batch_size, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def GetData(self):
        return self.data

    def GetBatches(self):
        return self.batches
