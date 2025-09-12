from odyssey.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, logger
from odyssey.data.trajectories_file import HDF5TrajectoriesFile
from odyssey.external.nle.populate_db import add_altorg_directory

import nle.dataset as nld
from nle.dataset.dataset import _ttyrec_generator, TtyrecDataset

import numpy as np
import os
import typer
import tqdm
import hashlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import List

FOLDER_NAMES = [
    "aa", "ab", "ac", "ad", "ae",
    "af", "ag", "ah", "ai", "aj",
    "ak", "al", "am", "an", "ao",
    "ap", "aq", "ar", "as", "at",
    "au", "av", "aw", "ax", "ay",
    "az", "ba", "bb", "bc", "bd",
    "be", "bf", "bg", "bh", "bi",
    "bj", "bk", "bl", "bm", "bn",
]

def get_ttyrecs_database(folder_names: List[str], data_dir: Path) -> TtyrecDataset:
    # Compute hash for storing the dataset
    hash_input = "".join(sorted(folder_names)).encode('utf-8')
    dataset_hash = hashlib.md5(hash_input).hexdigest()[:8]
    logger.info(f"Using folder hash: {dataset_hash}")

    # Check if the dataset path already exists
    db_file_path = INTERIM_DATA_DIR / f"ttyrecs_{dataset_hash}.db"
    if db_file_path.exists():
        logger.info(f"Found existing ttyrecs database '{db_file_path}'.")
    else:
        # Create the ttyrecs database
        nld.db.create(filename=str(db_file_path))
        logger.info(f"Creating ttyrecs database '{db_file_path}'")
        with tqdm.tqdm(folder_names, desc="Processing folders") as pbar:
            add_altorg_directory(path=str(data_dir), name="data", subfolders=folder_names, filename=str(db_file_path))
        logger.info(f"Successfully created ttyrecs database at '{db_file_path}'")
    
    return nld.TtyrecDataset("data", batch_size=1, seq_length=1, dbfilename=str(db_file_path))

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

def _create_hdf5_part(
    ttyrec_dataset: TtyrecDataset,
    game_ids: List[int],
    output_path: Path,
    chunk_size: int,
    position: int,
    compression_type: str = "gzip",
    keys=["tty_chars", "tty_colors", "tty_cursor"],
    batch_size: int = 1024,
):
    logger.info(f"Creating HDF5 dataset at '{output_path}' with {len(game_ids)} trajectories.")

    batch = next(create_ttyrec_generator(dataset=ttyrec_dataset, game_id=game_ids[0]))
    columns = {
        key: {"dtype": batch[key].dtype, "shape": batch[key].shape[2:]}
        for key in keys
    }

    dataset = HDF5TrajectoriesFile.create(file_path=output_path, columns=columns)

    def prep_batch(batch):
        curr_batch_size = (batch["gameids"] != 0).sum()
        data = np.zeros(curr_batch_size, dtype=dataset.dtype)
        for key in keys:
            data[key] = batch[key][0, :curr_batch_size]
        return data

    with tqdm.tqdm(game_ids, desc=f"Writing {output_path.name}", position=position) as pbar:
        for game_id in game_ids:
            paths = ttyrec_dataset.get_paths(game_id)
            player_name = Counter([Path(p).parent.name for p in paths]).most_common(1)[0][0].lower()
            metadata = {
                "dataset_timestamp": datetime.now(timezone.utc).isoformat(),
                "player_name": player_name,
                **ttyrec_dataset.get_meta(gameid=game_id),
            }

            generator = map(prep_batch, create_ttyrec_generator(ttyrec_dataset, game_id, seq_length=batch_size))
            dataset.add_trajectory_from_generator(
                name=str(game_id),
                batch_generator=generator,
                length_estimate=metadata.get("turns", 0) * 2,
                compression=compression_type,
                chunk_size=chunk_size,
                metadata=metadata
            )
            pbar.update(1)


app = typer.Typer()

@app.command()
def main(
    input_data_dir: Path = RAW_DATA_DIR / "nld_nao",
    output_path: Path = PROCESSED_DATA_DIR / "nld_nao_dataset.h5py",
    folder_names: List[str] = typer.Option(FOLDER_NAMES, "--files", "-f", help="List of folder names to include in the dataset."),
    chunk_size: int = 128,
    compression_type: str = "gzip"
):
    if output_path.exists():
        logger.info(f"Found existing HDF5 dataset at '{output_path}'. Skipping creation.")
        return
    
    logger.info(f"Creating ttyrecs database...")
    dataset = get_ttyrecs_database(folder_names, input_data_dir)

    output_parts_dir = output_path.parent / f"{output_path.stem}_parts"
    os.makedirs(output_parts_dir, exist_ok=True)

    num_workers = os.cpu_count() or 1
    game_ids = list(dataset._gameids)
    shard_ids = np.array_split(game_ids, num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _create_hdf5_part,
                ttyrec_dataset=dataset,
                game_ids=shard,
                chunk_size=chunk_size,
                compression_type=compression_type,
                output_path=output_parts_dir / f"part_{i+1}.hdf5",
                position=i
            )
            for i, shard in enumerate(shard_ids)
        ]

        logger.info(f"Generating dataset parts at '{output_parts_dir}' using {num_workers} workers...")
        for future in futures:
            future.result()

    logger.info(f"Combining parts into final HDF5 dataset '{output_path}'...")
    dataset_part_paths = [
        output_parts_dir / f"part_{i+1}.hdf5"
        for i in range(num_workers)
    ]
    HDF5TrajectoriesFile.combine(
        dataset_part_paths,
        output_path=output_path
    )

    logger.info(f"Successfully created HDF5 dataset at '{output_path}'.")

if __name__ == "__main__":
    app()