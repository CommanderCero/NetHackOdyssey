import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, TypedDict, Dict, Tuple, Union, Generator

def mixed_dtype_to_dict(data: np.ndarray):
    """
    Convert a mixed dtype numpy array to a dictionary.
    The keys are the field names and the values are the corresponding arrays.
    """
    if data.dtype.names is None:
        return {"data": data}

    return {
        name: data[name]
        for name in data.dtype.names
    }

class ColumnSpec(TypedDict):
    dtype: Union[np.dtype, type]
    shape: Tuple[int, ...]

class HDF5Trajectory:
    def __init__(self, data: h5py.Dataset):
        self.data = data

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        return mixed_dtype_to_dict(self.data[idx])

    def __len__(self):
        return len(self.data)

class HDF5TrajectoriesFile:
    def __init__(self,
        file: h5py.File,
        dtype: np.dtype=None,
    ):
        self.file = file
        self.dtype = dtype
        self.trajectory_names = list(file.keys())

        if dtype is None:
            first_trajectory = self.file[self.trajectory_names[0]]
            self.dtype = first_trajectory.dtype

    def add_trajectory_from_generator(
        self,
        name: str,
        batch_generator: Generator[np.ndarray, None, None],
        length_estimate: Optional[int] = None,
        compression='gzip',
        chunk_size=128,
        metadata: Optional[Dict] = None
    ):
        # Create a resizable dataset
        dset = self.file.create_dataset(
            name,
            shape=(length_estimate or 0,),
            maxshape=(None,),
            dtype=self.dtype,
            compression=compression,
            chunks=(chunk_size,)
        )

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                dset.attrs[key] = value

        # Fill the dataset in chunks
        offset = 0
        for batch in batch_generator:
            end = offset + len(batch)
            if end > len(dset):
                dset.resize((end * 2,))

            dset[offset:end] = batch
            offset = end

        # Resize to the actual length
        dset.resize((offset,))

    def __getitem__(self, name: str) -> HDF5Trajectory:
        if name not in self.file:
            raise KeyError(f"Trajectory '{name}' not found in the dataset.")
        return HDF5Trajectory(self.file[name])

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def close(self):
        self.file.close()

    @classmethod
    def load(self, file_path: str, mode="r") -> 'HDF5TrajectoriesFile':
        file = h5py.File(file_path, mode)
        return HDF5TrajectoriesFile(file)

    @classmethod
    def create(self, file_path: str, columns: Dict[str, ColumnSpec], mode="w") -> 'HDF5TrajectoriesFile':
        file = h5py.File(file_path, mode, libver='latest')
        dtype = np.dtype([(col, spec["dtype"], spec["shape"]) for col, spec in columns.items()])
        return HDF5TrajectoriesFile(file=file, dtype=dtype)

    @staticmethod
    def combine(
        part_paths: List[Path],
        output_path: Path
    ):
        with h5py.File(output_path, 'w', libver='latest') as vfile:
            for part_path in part_paths:
                relative_path = part_path.relative_to(output_path.parent)
                with h5py.File(part_path, 'r') as pfile:
                    for trajectory_name in pfile:
                        vfile[trajectory_name] = h5py.ExternalLink(str(relative_path), trajectory_name)
