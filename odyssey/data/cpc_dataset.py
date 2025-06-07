import torch
import torch.utils.data as data

import numpy as np

import h5py
import math
from typing import Callable, Optional

def mixed_dtype_to_dict(data: np.ndarray):
    """
    Convert a mixed dtype numpy array to a dictionary.
    The keys are the field names and the values are the corresponding arrays.
    """
    if data.dtype.names is None:
        return data
    
    return {
        name: data[name]
        for name in data.dtype.names
    }

class CPCDataset(data.IterableDataset):
    """
    Dataset for Contrastive Predictive Coding (CPC) using HDF5 files.
    This dataset samples context in form of a sequence of observations.
    The context is used to predict a future observation called the positive sample.
    This dataset does not sample negative samples, as the other positive samples in the batch should be used as the negative samples.

    The HDF5 file should store each trajectory as a dataset in the root group.
    Each trajectory should only store the ordered sequence of observations.
    In case the observation consists of multiple fields, they should be stored as a structured array (recarray).
    It is recommended to chunk each dataset, ensuring that the chunk size is larger than the context length + future length.
    """
    def __init__(self,
        h5py_file_path: str,
        batch_size: int,
        context_length: int,
        future_length: int,
        samples_per_trajectory: int,
        seed: int = None,
        transform: Optional[Callable]=None
    ):
        """
        Args:
            h5py_file_path (str): Path to the HDF5 file.
            batch_size (int): Number of samples per batch, use batch_size=None when using a DataLoader with this dataset.
            context_length (int): Maximum length of the context sequence.
            future_length (int): Maximum offset into the future for sampling a positive observation.
            samples_per_trajectory (int): Number of samples per trajectory. Having more samples of the same trajectory in a batch should increase the amount of difficult negative samples.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            transform (Callable, optional): Optional transform to be applied on the data. Defaults to None.
        """
        super().__init__()
        self.data = h5py.File(h5py_file_path, 'r')
        self.batch_size = batch_size
        self.context_length = context_length
        self.future_length = future_length
        self.samples_per_trajectory = samples_per_trajectory
        self.seed = seed
        self.transform = transform

        self.valid_trajectory_keys = [
            key
            for key in self.data.keys()
            if len(self.data[key]) > self.samples_per_trajectory + 1
        ]
        assert len(self.valid_trajectory_keys) > 0, f"No trajectories found with a length > samples_per_trajectory + 1 ({self.samples_per_trajectory + 1}). Reduce the amount of samples, as otherwise a batch would contain the same positive sample multiple times."

        self.data_dtype = self.data[self.valid_trajectory_keys[0]].dtype
        self.data_shape = self.data[self.valid_trajectory_keys[0]].shape

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        while True:
            X = np.zeros((self.batch_size, self.context_length, *self.data_shape[1:]), dtype=self.data_dtype)
            X_padding_mask = np.zeros((self.batch_size, self.context_length), dtype=bool)
            y = np.zeros((self.batch_size, *self.data_shape[1:]), dtype=self.data_dtype)
            y_indices = np.zeros((self.batch_size,), dtype=int)

            slices = self.generate_batch_slices(rng)
            for i, (key, start, end, offset) in enumerate(slices):
                context_length = end - start
                data_slice = self.data[key][start:end+offset+1] # Load only once, which should be more efficient if the used chunk size is larger than context_length + future_length.

                X[i, :context_length] = data_slice[:context_length]
                X_padding_mask[i, context_length:] = True

                y[i] = data_slice[context_length+offset]
                y_indices[i] = offset

            data = {
                "context": mixed_dtype_to_dict(X),
                "padding_mask": X_padding_mask,
                "positive_samples": mixed_dtype_to_dict(y),
                "positive_indices": y_indices
            }

            if self.transform is not None:
                data = self.transform(data)
            
            yield data

    def generate_batch_slices(self, rng: np.random.Generator):
        """
        This function samples a batch of slices, ensuring no positive sample is returned more than once per batch.
        Each slice is a tuple of (trajectory_key, start, end, offset).
        The start and end indices are used to gather the context.
        The offset is used to determine the positive sample (end + offset).
        Note that end is not inclusive, meaning when offset is 0, the positive index equals end
        """
        num_key_samples = math.ceil(self.batch_size / self.samples_per_trajectory)
        trajectory_keys = rng.choice(self.valid_trajectory_keys, num_key_samples, replace=False)
        slices = []

        for key in trajectory_keys:
            num_samples = min(self.batch_size - len(slices), self.samples_per_trajectory)
            positive_indices = rng.choice(range(1, len(self.data[key])), num_samples, replace=False)
            offsets = rng.integers(0, np.minimum(positive_indices, self.future_length))
            ends = positive_indices - offsets
            starts = np.maximum(ends - self.context_length, 0)

            slices.extend(
                (key, start, end, offset)
                for start, end, offset in zip(starts, ends, offsets)
            )

        return slices

if __name__ == "__main__":
    dataset = CPCDataset(
        h5py_file_path="/workspace/data/nld_nao_train.h5",
        batch_size=33,
        context_length=100,
        future_length=30,
        samples_per_trajectory=8
    )
    
    slices = dataset.generate_batch_slices()

    from torch.utils.data import DataLoader        
    import time

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=2
    )
    
    start_time = time.time()
    num_batches = 100

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

    elapsed_time = time.time() - start_time
    print(f"Fetched {num_batches} batches in {elapsed_time:.2f} seconds.")



