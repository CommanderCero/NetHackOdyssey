import nle.dataset as nld
import h5py
import numpy as np
import argparse
import tqdm

def load_game_data(dataset: nld.TtyrecDataset, game_id: int, load_keys=["tty_chars", "tty_colors", "tty_cursor"]) -> dict:
    # The last step is always a padding step, so we remove it
    steps = dataset.get_ttyrec(game_id, 1)[:-1]
    assert all(step["gameids"][0, 0] == game_id for step in steps), f"Game ID mismatch"

    data = {
        key: np.stack([step[key].squeeze() for step in steps])
        for key in load_keys
    }
    dtype = np.dtype([
        (key, arr.dtype, arr.shape[1:])
        for key, arr in data.items()
    ])

    return np.rec.fromarrays(data.values(), names=list(data.keys()), dtype=dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--altorg_dir", type=str, default="/workspace/data/nld_nao")
    parser.add_argument("--output_h5_path", type=str, default="/workspace/data/nld_nao.h5")
    parser.add_argument("--compression_type", type=str, default="gzip")
    parser.add_argument("--chunk_size", type=int, default=128)
    args = parser.parse_args()

    # Create/Load the database
    if not nld.db.exists():
        nld.db.create()
        nld.add_altorg_directory(args.altorg_dir, "nld-nao-v0")
    dataset = nld.TtyrecDataset("nld-nao-v0", batch_size=1, seq_length=1)

    # Create the HDF5 file
    with h5py.File(args.output_h5_path, "w") as file:
        for game_id in tqdm.tqdm(dataset._gameids, desc=f"Generating h5 file"):
            metadata = dataset.get_meta(game_id)
            game_data = load_game_data(dataset, game_id)

            chunk_size = min(args.chunk_size, len(game_data))
            game_dataset = file.create_dataset(
                name=str(game_id),
                data=game_data,
                compression=args.compression_type,
                chunks=(chunk_size,),
                maxshape=(None),
            )

            game_dataset.attrs.update(metadata)