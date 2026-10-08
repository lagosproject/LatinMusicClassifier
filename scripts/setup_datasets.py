import os
import shutil
import argparse
import kagglehub
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_and_copy(dataset_handle, target_filename, source_filename=None):
    """
    Downloads a dataset from Kaggle and copies a specific file to the current directory.
    """
    try:
        logger.info(f"Downloading dataset: {dataset_handle}...")
        path = kagglehub.dataset_download(dataset_handle)
        logger.info(f"Dataset downloaded to: {path}")

        # If source_filename is not provided, use target_filename
        if not source_filename:
            source_filename = target_filename

        # Search for the file in the downloaded path
        source_path = None
        for root, dirs, files in os.walk(path):
            if source_filename in files:
                source_path = os.path.join(root, source_filename)
                break
        
        if source_path:
            logger.info(f"Found {source_filename} at {source_path}")
            shutil.copy(source_path, target_filename)
            logger.info(f"✅ Copied to {target_filename}")
            return True
        else:
            logger.error(f"❌ Could not find {source_filename} in the downloaded dataset.")
            # List files to help debug
            all_files = []
            for root, dirs, files in os.walk(path):
                all_files.extend(files)
            logger.info(f"Available files: {all_files}")
            return False

    except Exception as e:
        logger.error(f"❌ Error downloading {dataset_handle}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle.")
    parser.add_argument("--with-cookies", action="store_true", help="Also attempt to download YouTube cookies from Kaggle (requires Kaggle authentication and access)")
    args = parser.parse_args()

    datasets = [
        {
            "handle": "g5fighter/playlist-latin-music",
            "target": os.path.join("data", "spotify_playlists_dataset.csv")
        },
        {
            "handle": "g5fighter/latin-music-embeddings",
            "target": os.path.join("data", "latin_music_features_dataset.csv")
        },
    ]

    if args.with_cookies:
        datasets.append({
            "handle": "g5fighter/youtube-cookies",
            "target": os.path.join("config", "www.youtube.com_cookies.txt"),
            "source": "www.youtube.com_cookies.txt"
        })

    success_count = 0
    for ds in datasets:
        if download_and_copy(ds["handle"], ds["target"], ds.get("source")):
            success_count += 1

    logger.info(f"\nSetup complete. {success_count}/{len(datasets)} datasets processed successfully.")

if __name__ == "__main__":
    main()
