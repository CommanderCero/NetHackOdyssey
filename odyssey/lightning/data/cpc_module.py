from odyssey.data.cpc_dataset import CPCDataset

import torch
from lightning import LightningDataModule

import nle.dataset as nld
from nle.dataset.dataset import _ttyrec_generator, TtyrecDataset

import numpy as np
import requests
import tempfile
import zipfile
import os
import h5py
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

NLD_NAO_BASE_URL = "https://dl.fbaipublicfiles.com/nld/nld-nao"
NLD_NAO_SUFFIXES = [
    "dir-aa", "dir-ab", "dir-ac", "dir-ad", "dir-ae", "dir-af", "dir-ag", "dir-ah", "dir-ai", "dir-aj",
    "dir-ak", "dir-al", "dir-am", "dir-an", "dir-ao", "dir-ap", "dir-aq", "dir-ar", "dir-as", "dir-at",
    "dir-au", "dir-av", "dir-aw", "dir-ax", "dir-ay", "dir-az", "dir-ba", "dir-bb", "dir-bc", "dir-bd",
    "dir-be", "dir-bf", "dir-bg", "dir-bh", "dir-bi", "dir-bj", "dir-bk", "dir-bl", "dir-bm", "dir-bn",
    "xlogfiles"
]

def download_and_extract(suffix, output_dir, position):
    url = f"{NLD_NAO_BASE_URL}/nld-nao-{suffix}.zip"
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file, \
             tqdm(total=total, unit='B', unit_scale=True, desc=f"{suffix}", position=position, leave=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    pbar.update(len(chunk))
            tmp_file_path = tmp_file.name

    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(tmp_file_path)

def download_nld_nao_datasets(output_dir, suffixes=NLD_NAO_SUFFIXES):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_and_extract, suffix, output_dir, idx)
            for idx, suffix in enumerate(suffixes)
        ]
        for future in futures:
            future.result()

def create_ttyrec_generator(
    dataset: TtyrecDataset,
    game_id: int,
    batch_size: int = 1,
    seq_length: int = 1
):
    """
    Creates an ttyrec generator for a single game.
    Avoids the out-of-memory errors with dataset.get_ttyrec which collects all batches at once.
    """
    load_fn = dataset._make_load_fn([game_id])
    iter = _ttyrec_generator(
        batch_size=batch_size,
        seq_length=seq_length,
        rows=dataset.rows,
        cols=dataset.cols,
        load_fn=load_fn,
        map_fn=dataset._map,
        ttyrec_version=dataset._ttyrec_version,
    )
    return iter

def add_game_to_h5(
    dataset: TtyrecDataset,
    file: h5py.File,
    game_id: int,
    chunk_size: int,
    compression_type: str = "gzip",
    keys=["tty_chars", "tty_colors", "tty_cursor"],
    batch_size: int = 1024
) -> h5py.Dataset:
    """
    Adds a single game as a dataset to an HDF5 file.
    Returns the created dataset.

    Implemented very weirdly, but it works (I think).
    """
    # Use example batch to determine dtype
    batch = next(create_ttyrec_generator(dataset=dataset, game_id=game_id))
    dtype = np.dtype([
        (key, arr.dtype, arr.shape[2:]) # Shape without batch_size and sequence_length
        for key, arr in batch.items()
        if key in keys
    ])

    def prep_batch(batch):
        # Compute batch_size as the ttyrec_generator adds padding at the end
        curr_batch_size = (batch["gameids"] != 0).sum()

        data = np.zeros(curr_batch_size, dtype=dtype)
        for key in keys:
            data[key] = batch[key][0, :curr_batch_size]
        
        return data

    # Initialize dataset with resizeable shape
    # We use the number of turns as an estimate for how much space we need
    turns = dataset.get_meta(game_id)["turns"]
    ds = file.create_dataset(
        name=str(game_id),
        shape=(turns,),
        dtype=dtype,
        compression=compression_type,
        chunks=(chunk_size,),
        maxshape=(None,)
    )

    # Add batches
    offset = 0
    for batch in create_ttyrec_generator(dataset=dataset, game_id=game_id, seq_length=batch_size):
        batch = prep_batch(batch)
        end = offset + batch.shape[0]
        if end > ds.shape[0]:
            ds.resize((end + turns,))
        ds[offset:end] = batch
        offset = end
    ds.resize((offset,))  # Resize to the actual size

    return ds

def write_games_to_h5(game_ids, dataset: nld.TtyrecDataset, output_path, chunk_size, compression_type):
    with h5py.File(output_path, "w") as file:
        for game_id in game_ids:
            ds = add_game_to_h5(
                dataset=dataset,
                file=file,
                game_id=game_id,
                chunk_size=chunk_size,
                compression_type=compression_type
            )
            metadata = dataset.get_meta(game_id)
            ds.attrs.update(dict(metadata))
            # Add player name using paths (As metadata.name (ingame name) might be different from the player name)
            paths = dataset.get_paths(game_id)
            names = [Path(path).parent.name for path in paths]
            assert len(set(names)) == 1, "Found multiple names for same player"
            ds.attrs["player_name"] = names[0]


def _write_part(part_id, part_game_ids, dataset_path, output_dir, prefix, chunk_size, compression_type):
    dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1, seq_length=1, dbfilename=dataset_path)
    out_path = os.path.join(output_dir, f"{prefix}_part{part_id}.h5")
    write_games_to_h5(
        part_game_ids,
        dataset,
        out_path,
        chunk_size,
        compression_type
    )
    return out_path

def write_games_to_h5_parallel(game_ids, dataset, output_dir, prefix, chunk_size, compression_type, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = dataset.dbfilename
    shard_ids = np.array_split(game_ids, num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _write_part,
                idx,
                shard.tolist(),
                dataset_path,
                output_dir,
                prefix,
                chunk_size,
                compression_type
            )
            for idx, shard in enumerate(shard_ids) if len(shard) > 0
        ]
        part_paths = [f.result() for f in futures]
    
    return part_paths

def create_virtual_dataset(vds_path, part_paths):
    with h5py.File(vds_path, 'w', libver='latest') as vfile:
        for part_path in part_paths:
            relative_path = os.path.relpath(part_path, os.path.dirname(vds_path))
            with h5py.File(part_path, 'r') as part_file:
                for game_id in part_file:
                    vfile[game_id] = h5py.ExternalLink(relative_path, f"/{game_id}")

class CPCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        context_length: int = 100,
        future_length: int = 30,
        batch_size: int = 32,
        num_test_samples: int = 50,
        samples_per_trajectory: int = 10,
        num_workers: int = 0,
        nld_nao_suffixes = NLD_NAO_SUFFIXES
    ):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        if len(os.listdir(self.raw_data_dir)) != 0:
            print(f"Data directory {self.raw_data_dir} already contains files. Skipping download.")
        else:
            download_nld_nao_datasets(self.raw_data_dir, suffixes=self.hparams.nld_nao_suffixes)

        if os.path.exists(self.db_file):
            print(f"Database file {self.db_file} already exists. Skipping database creation.")
        else:
            print(f"Creating database file {self.db_file}...")
            nld.db.create(filename=self.db_file)
            nld.add_altorg_directory(path=self.raw_data_dir, name="nld-nao-v0", filename=self.db_file)

        if os.path.exists(self.train_file) and os.path.exists(self.test_file):
            print(f"Train file {self.train_file} and test file {self.test_file} already exist. Skipping HDF5 creation.")
        else:
            print(f"Creating HDF5 files {self.train_file} and {self.test_file}...")
            dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1, seq_length=1, dbfilename=self.db_file)
            game_ids = list(dataset._gameids)
            random.shuffle(game_ids)

            test_ids = game_ids[:self.hparams.num_test_samples]
            train_ids = game_ids[self.hparams.num_test_samples:]

            chunk_size = (self.hparams.future_length + self.hparams.context_length) * 2
            
            # Parallelize geenerating the train dataset
            print(f"Generating {self.train_file}...")
            train_parts = write_games_to_h5_parallel(
                train_ids, dataset, os.path.join(self.hparams.data_dir, "train_parts"), "train",
                chunk_size=chunk_size, compression_type="gzip"
            )
            create_virtual_dataset(self.train_file, train_parts)

            # Test data shouldnt be much, so we do not parallelize it
            print(f"Generating {self.test_file}...")
            write_games_to_h5(test_ids, dataset, self.test_file, chunk_size, "gzip")

    def train_dataloader(self):
        dataset = CPCDataset(
            self.train_file,
            batch_size=self.hparams.batch_size,
            context_length=self.hparams.context_length,
            future_length=self.hparams.future_length,
            samples_per_trajectory=self.hparams.samples_per_trajectory,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        dataset = CPCDataset(
            self.test_file,
            batch_size=self.hparams.batch_size,
            context_length=self.hparams.context_length,
            future_length=self.hparams.future_length,
            samples_per_trajectory=self.hparams.samples_per_trajectory,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    @property
    def db_file(self):
        return os.path.join(self.hparams.data_dir, "ttyrecs.db")
    
    @property
    def raw_data_dir(self):
        return os.path.join(self.hparams.data_dir, "nld_nao")
    
    @property
    def train_file(self):
        return os.path.join(self.hparams.data_dir, "nld_nao_train.h5")
    
    @property
    def test_file(self):
        return os.path.join(self.hparams.data_dir, "nld_nao_test.h5")

    
if __name__ == "__main__":
    data_module = CPCDataModule(
        data_dir="/workspace/data",
        context_length=100,
        future_length=30,
        batch_size=32,
        num_test_samples=50,
        samples_per_trajectory=10,
        num_workers=0
    )

    data_module.prepare_data()

    for batch in data_module.train_dataloader():
        print(batch)
        break