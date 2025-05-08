import nle.dataset as nld
import h5py
import numpy as np
import argparse
import tqdm
import random

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

def write_games_to_h5(game_ids, dataset, output_path, chunk_size, compression_type, desc):
    with h5py.File(output_path, "w") as file:
        for game_id in tqdm.tqdm(game_ids, desc=desc):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--altorg_dir", type=str, default="/workspace/data/nld_nao")
    parser.add_argument("--train_output_h5", type=str, default="/workspace/data/nld_nao_train.h5")
    parser.add_argument("--test_output_h5", type=str, default="/workspace/data/nld_nao_test.h5")
    parser.add_argument("--compression_type", type=str, default="gzip")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--test_samples", type=int, default=50, help="Number of test samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Initialize dataset
    if not nld.db.exists():
        nld.db.create()
        nld.add_altorg_directory(args.altorg_dir, "nld-nao-v0")
    dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1, seq_length=1)

    # Shuffle and split
    game_ids = list(dataset._gameids)
    random.seed(args.seed)
    random.shuffle(game_ids)

    test_ids = game_ids[:args.test_samples]
    train_ids = game_ids[args.test_samples:]

    # Write train and test HDF5 files
    write_games_to_h5(train_ids, dataset, args.train_output_h5, args.chunk_size, args.compression_type, "Writing train set")
    write_games_to_h5(test_ids, dataset, args.test_output_h5, args.chunk_size, args.compression_type, "Writing test set")
