from odyssey.data.cpc_dataset import CPCDataset

import torch
from lightning import LightningDataModule

import nle.dataset as nld

import numpy as np
import requests
import tempfile
import zipfile
import os
import h5py
import random
from tqdm import tqdm

NLD_NAO_BASE_URL = "https://dl.fbaipublicfiles.com/nld/nld-nao"
NLD_NAO_SUFFIXES = [
    "dir-aa", "dir-ab", "dir-ac", "dir-ad", "dir-ae", "dir-af", "dir-ag", "dir-ah", "dir-ai", "dir-aj",
    "dir-ak", "dir-al", "dir-am", "dir-an", "dir-ao", "dir-ap", "dir-aq", "dir-ar", "dir-as", "dir-at",
    "dir-au", "dir-av", "dir-aw", "dir-ax", "dir-ay", "dir-az", "dir-ba", "dir-bb", "dir-bc", "dir-bd",
    "dir-be", "dir-bf", "dir-bg", "dir-bh", "dir-bi", "dir-bj", "dir-bk", "dir-bl", "dir-bm", "dir-bn",
    "xlogfiles"
]

def download_nld_nao_datasets(output_dir, suffixes=NLD_NAO_SUFFIXES):
    os.makedirs(output_dir, exist_ok=True)

    for suffix in suffixes:
        url = f"{NLD_NAO_BASE_URL}/nld-nao-{suffix}.zip"
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file, \
                 tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {url}...") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        pbar.update(len(chunk))
                tmp_file_path = tmp_file.name

        print(f"Extracting nld-nao-{suffix}.zip to {output_dir}...")
        with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(tmp_file_path)

def load_game_data(dataset: nld.TtyrecDataset, game_id: int, load_keys=["tty_chars", "tty_colors", "tty_cursor"]) -> dict:
    steps = dataset.get_ttyrec(game_id, 1)[:-1]
    assert all(step["gameids"][0, 0] == game_id for step in steps), "Game ID mismatch"

    data = {
        key: np.stack([step[key].squeeze() for step in steps])
        for key in load_keys
    }
    dtype = np.dtype([
        (key, arr.dtype, arr.shape[1:])
        for key, arr in data.items()
    ])
    return np.rec.fromarrays(data.values(), names=list(data.keys()), dtype=dtype)

def write_games_to_h5(game_ids, dataset: nld.TtyrecDataset, output_path, chunk_size, compression_type, desc):
    with h5py.File(output_path, "w") as file:
        for game_id in tqdm(game_ids, desc=desc):
            metadata = dataset.get_meta(game_id)
            game_data = load_game_data(dataset, game_id)
            chunk_size = min(chunk_size, len(game_data))

            ds = file.create_dataset(
                name=str(game_id),
                data=game_data,
                compression=compression_type,
                chunks=(chunk_size,),
                maxshape=(None,)
            )
            ds.attrs.update(metadata)

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
            write_games_to_h5(train_ids, dataset, self.train_file, chunk_size, "gzip", "Writing train set")
            write_games_to_h5(test_ids, dataset, self.test_file, chunk_size, "gzip", "Writing test set")

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
    
    def _prepare_datasets(self):
        nld.db.create()
        nld.add_altorg_directory(self.data_dir, "nld-nao-v0")

        dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1, seq_length=1)
        game_ids = list(dataset._gameids)
        random.seed(self.seed)
        random.shuffle(game_ids)

        test_ids = game_ids[:self.test_samples]
        train_ids = game_ids[self.test_samples:]

        write_games_to_h5(train_ids, dataset, self.train_output_h5, self.chunk_size, self.compression_type, "Writing train set")
        write_games_to_h5(test_ids, dataset, self.test_output_h5, self.chunk_size, self.compression_type, "Writing test set")

    
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
