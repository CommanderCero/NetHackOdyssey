from odyssey.config import RAW_DATA_DIR, logger

import typer
import requests
import zipfile
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from typing import List

XLOGFILES_URL = "https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao_xlogfiles.zip"
VALID_FILE_NAMES = [
    "aa", "ab", "ac", "ad", "ae",
    "af", "ag", "ah", "ai", "aj",
    "ak", "al", "am", "an", "ao",
    "ap", "aq", "ar", "as", "at",
    "au", "av", "aw", "ax", "ay",
    "az", "ba", "bb", "bc", "bd",
    "be", "bf", "bg", "bh", "bi",
    "bj", "bk", "bl", "bm", "bn",
]
URL_FORMAT = "https://dl.fbaipublicfiles.com/nld/nld-nao/nld-nao-dir-{file_name}.zip"

def download_and_extract_zip(url, output_path, position, description=None):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file, \
             tqdm(total=total, unit='B', unit_scale=True, desc=description, position=position, leave=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                pbar.update(len(chunk))
            tmp_file_path = tmp_file.name

    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(tmp_file_path)

def download_nao_folder(name: str, output_path: Path, position: int):
    url = URL_FORMAT.format(file_name=name)
    download_and_extract_zip(url, output_path, position, description=f"Downloading {name}")

    # Each zip file contains a subfolder named nld-nao-unzipped
    os.rename(output_path / "nld-nao-unzipped", output_path / name)

def download_xlogfiles(output_path: Path, position: int):
    download_and_extract_zip(XLOGFILES_URL, output_path, position, description="Downloading xlogfiles")

app = typer.Typer()

@app.command()
def main(
    files: List[str] = typer.Option(VALID_FILE_NAMES, "--files", "-f", help="List of files to download. Defaults to downloading all files."),
    output_path: Path = RAW_DATA_DIR / "nld_nao",
    force: bool = False
):
    invalid_files = list(set(files) - set(VALID_FILE_NAMES))
    if invalid_files:
        logger.error(f"Invalid file names: {invalid_files}. Valid names are: {VALID_FILE_NAMES}")
        return -1
    
    os.makedirs(output_path, exist_ok=True)

    # Skip already downloaded files
    if not force:
        for file in files:
            if (output_path / file).exists():
                logger.info(f"Directory '{output_path / file}' already exists, skipping.")
                files.remove(file)

    # Download files in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_nao_folder, suffix, output_path, idx)
            for idx, suffix in enumerate(files)
        ]

        if (output_path / "xlogfile.full.txt").exists() and not force:
            logger.info(f"Directory '{output_path}' already contains xlogfiles, skipping.")
        else:
            futures.append(executor.submit(download_xlogfiles, output_path, len(files)))

        for future in futures:
            future.result()

    logger.info("Download and extraction complete.")
    return 0

if __name__ == "__main__":
    app()
