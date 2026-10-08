import pandas as pd
import os
import logging

# ── Configuration & Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Paths
    playlist_file = os.path.join("data", "spotify_playlists_dataset.csv")
    latin_features_file = os.path.join("data", "latin_music_features_dataset.csv")
    subdivided_features_file = os.path.join("data", "subdivided_playlist_features.csv")

    # 1. Load ground truth mapping (song_id -> genre)
    if not os.path.exists(playlist_file):
        logger.error(f"Playlist file not found: {playlist_file}")
        return
    
    playlist_df = pd.read_csv(playlist_file)
    id_to_genre = dict(zip(playlist_df['song_id'], playlist_df['genre']))
    playlist_ids = set(playlist_df['song_id'])
    logger.info(f"Loaded {len(playlist_ids)} unique songs from playlist dataset.")

    # 2. Identify missing songs in the main features dataset
    if not os.path.exists(latin_features_file):
        logger.warning(f"Latin features file not found. Creating a new one.")
        existing_ids = set()
        latin_df = pd.DataFrame()
    else:
        # Load only song_id to save memory
        latin_df_ids = pd.read_csv(latin_features_file, usecols=['song_id'])
        existing_ids = set(latin_df_ids['song_id'].unique())
    
    missing_ids = playlist_ids - existing_ids
    logger.info(f"Found {len(missing_ids)} songs missing from latin_music_features_dataset.")

    if not missing_ids:
        logger.info("No missing songs to migrate.")
        return

    # 3. Check for these songs in subdivided_playlist_features.csv
    if not os.path.exists(subdivided_features_file):
        logger.info(f"Subdivided features file not found: {subdivided_features_file}. Nothing to move.")
        return

    logger.info(f"Loading subdivided features from {subdivided_features_file}...")
    sub_df = pd.read_csv(subdivided_features_file)
    
    # Filter for missing IDs
    to_move_mask = sub_df['song_id'].isin(missing_ids)
    to_move_df = sub_df[to_move_mask].copy()
    
    if to_move_df.empty:
        logger.info("None of the missing songs were found in subdivided_playlist_features.csv.")
        return

    moved_count = len(to_move_df['song_id'].unique())
    logger.info(f"Identified {moved_count} songs ({len(to_move_df)} fragments) to migrate.")

    # 4. Tag correctly by genre and reorder columns
    # Re-tag using ground truth
    to_move_df['genre'] = to_move_df['song_id'].map(id_to_genre)
    
    # Column order in subdivided: genre, song_id, fragment_index, bpm, fragment_start_sec, ...
    # Column order in latin:     genre, song_id, fragment_index, fragment_start_sec, bpm, ...
    
    cols = list(to_move_df.columns)
    # Swap bpm (index 3) and fragment_start_sec (index 4)
    # subdivided order: ['genre', 'song_id', 'fragment_index', 'bpm', 'fragment_start_sec', ...]
    # target order:     ['genre', 'song_id', 'fragment_index', 'fragment_start_sec', 'bpm', ...]
    
    # Verify column names at these indices to be safe
    if cols[3] == 'bpm' and cols[4] == 'fragment_start_sec':
        cols[3], cols[4] = cols[4], cols[3]
        to_move_df = to_move_df[cols]
        logger.info("Successfully reordered columns (swapped bpm and fragment_start_sec).")
    else:
        logger.warning(f"Column mismatch at expected indices. Expected [3]=bpm, [4]=fragment_start_sec. Found [{cols[3]}], [{cols[4]}]. Skipping reorder.")

    # 5. Append to latin_music_features_dataset.csv
    to_move_df.to_csv(latin_features_file, mode='a', header=not os.path.exists(latin_features_file), index=False)
    logger.info(f"Successfully appended {len(to_move_df)} rows to {latin_features_file}.")

    # 6. Remove migrated rows from subdivided_playlist_features.csv
    remaining_sub_df = sub_df[~to_move_mask]
    remaining_sub_df.to_csv(subdivided_features_file, index=False)
    logger.info(f"Successfully removed migrated rows from {subdivided_features_file}. ({len(remaining_sub_df)} rows remaining)")

if __name__ == "__main__":
    main()
