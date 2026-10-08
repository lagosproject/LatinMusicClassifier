import os
import re
import shutil
import glob
import subprocess
import argparse
import time
import logging
import unicodedata
from typing import Dict, List, Optional, Set

import torch
import librosa
import numpy as np
import pandas as pd
from transformers import ClapModel, ClapProcessor
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm
import yt_dlp

# ── Configuration & Logging ──────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "dataset_generator.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Checks if ffmpeg is available in the system."""
    if not shutil.which("ffmpeg"):
        logger.error("❌ FFmpeg not found! Please install it and add it to your PATH.")
        logger.error("Download from: https://www.gyan.dev/ffmpeg/builds/")
        return False
    return True

# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercases, removes accents, and removes non-alphanumeric characters for robust comparison."""
    if not text:
        return ""
    # Decompose accented characters (e.g., 'é' -> 'e' + accent)
    text = unicodedata.normalize('NFKD', str(text))
    # Remove the accent marks (non-spacing marks)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    # Lowercase and remove symbols/spaces
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

def get_node_path():
    """Finds the node executable path."""
    path = shutil.which("node")
    if not path:
        # Check common windows locations if not in PATH
        common_paths = [
            "C:\\Program Files\\nodejs\\node.exe",
            "C:\\Program Files (x86)\\nodejs\\node.exe"
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
    return path

def _ydl_info_opts(cookies_file: Optional[str] = None) -> dict:
    """Common yt-dlp options for metadata extraction."""
    node_path = get_node_path()
    opts = {
        'quiet': True,
        'no_warnings': True,
        'sleep_interval': 5,          # Delay between requests
        'max_sleep_interval': 15,     # Max delay
        'socket_timeout': 30,
        'nocheckcertificate': True,
        'format': 'bestaudio/best',
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'mweb', 'tv'],
            }
        },
        'remote_components': ['ejs:github'],
    }
    if node_path:
        opts['js_runtimes'] = {'node': {'path': node_path}}
    
    target_cookies = cookies_file or os.path.join('config', 'www.youtube.com_cookies.txt')
    if target_cookies and os.path.exists(target_cookies):
        opts['cookiefile'] = target_cookies
    return opts

def search_official_video(track: str, artist: Optional[str] = None, song_id: Optional[str] = None, cookies_file: Optional[str] = None) -> Optional[dict]:
    """Finds the best YouTube video for a song using a 3-tier strategy."""
    log_prefix = f"[{song_id}] {track} - {artist}" if song_id else f"{track} - {artist}"
    query = track
    
    strategies = [
        (f'ytsearch5:{query} - Topic',       'Topic channel'),
        (f'ytsearch5:{query} official audio', 'Official audio'),
        (f'ytsearch5:{query}',                'Plain search'),
    ]
    
    if artist:
        strategies.extend([
            (f'ytsearch5:{query} {artist} - Topic',       'Topic channel (with artist)'),
            (f'ytsearch5:{query} {artist} official audio', 'Official audio (with artist)'),
            (f'ytsearch5:{query} {artist}',                'Plain search (with artist)'),
        ])

    search_logs = []

    with yt_dlp.YoutubeDL(_ydl_info_opts(cookies_file)) as ydl:
        for search_query, strategy_name in strategies:
            try:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get('entries', [])
                if not entries:
                    search_logs.append(f"      [Strategy: {strategy_name}] No entries found.")
                    continue

                search_logs.append(f"      [Strategy: {strategy_name}] Found {len(entries)} entries:")
                for idx, entry in enumerate(entries):
                    uploader = (entry.get('uploader') or "").lower()
                    channel  = (entry.get('channel') or "").lower()
                    title    = (entry.get('title') or "").lower()
                    
                    search_logs.append(f"        {idx+1}. Title: '{entry.get('title')}' | Uploader: '{entry.get('uploader')}'")

                    uploader_norm = normalize_text(uploader)
                    channel_norm  = normalize_text(channel)
                    title_norm    = normalize_text(title)
                    artist_norm   = normalize_text(artist) if artist else ""

                    if strategy_name == 'Topic channel':
                        if channel.strip().endswith('- topic') and (not artist or artist.lower() in channel or artist_norm in channel_norm):
                            logger.info(f'   -> {log_prefix} | Topic channel match: {entry.get("channel")}')
                            return entry
                        # If it's a topic channel but doesn't match artist, we still might want it as fallback
                    
                    blacklist = [' tv', 'news', 'interview', 'entrevista', 'show']
                    if any(word in uploader for word in blacklist):
                        if artist and artist.lower() not in uploader and artist_norm not in uploader_norm:
                            search_logs.append(f"          -> Skipped (Blacklisted term in uploader)")
                            continue
                        
                    if not artist or (artist.lower() in uploader or artist.lower() in title or 
                                      artist_norm in uploader_norm or artist_norm in title_norm):
                        logger.info(f'   -> {log_prefix} | Found candidate ({strategy_name}): {entry.get("uploader") or entry.get("channel")}')
                        return entry
                    else:
                        search_logs.append(f"          -> Skipped (Artist '{artist}' not matched in uploader '{uploader}' or title '{title}')")
                
                # If we found a good match in the first strategy, don't bother with others
                if strategy_name == 'Topic channel' and entries:
                    # We already checked them above, if we are here it's because none matched perfectly.
                    # But if we found ANY topic channel, it's probably better than "official audio" search.
                    pass

            except Exception as e:
                logger.warning(f'   -> {log_prefix} | Error in strategy "{strategy_name}": {e}')
                search_logs.append(f"      [Strategy: {strategy_name}] Error: {e}")
                
    # If we get here, no video was returned. Print the accumulated log.
    logger.warning(f"   -> ⚠️ {log_prefix} | Search failed. Search Details:")
    for log_msg in search_logs:
        logger.warning(f"      {log_prefix} | {log_msg.strip()}")
        
    return None

def select_fragment_times(
    heatmap: Optional[List[Dict]],
    total_duration: float,
    n_fragments: int,
    fragment_duration: float,
    min_distance: float,
) -> List[float]:
    """Chooses N start times for audio fragments based on heatmap or uniform spacing."""
    max_start = total_duration - fragment_duration
    if max_start <= 0:
        return [0.0]

    if heatmap:
        candidates = sorted(heatmap, key=lambda x: x.get('value', 0), reverse=True)
        selected = []
        for segment in candidates:
            t = float(segment.get('start_time', 0))
            t = max(0.0, min(t, max_start))
            if not any(abs(t - t_prev) < min_distance for t_prev in selected):
                selected.append(t)
            if len(selected) == n_fragments:
                break
        if selected:
            return sorted(selected)

    # Fallback: uniform spacing (10% to 90%)
    zone_start = total_duration * 0.10
    zone_end = max(total_duration * 0.90 - fragment_duration, zone_start)
    if n_fragments == 1:
        return [zone_start + (zone_end - zone_start) / 2]

    step = (zone_end - zone_start) / (n_fragments - 1) if n_fragments > 1 else 0
    times = [zone_start + i * step for i in range(n_fragments)]
    filtered = [times[0]]
    for t in times[1:]:
        if t - filtered[-1] >= min_distance:
            filtered.append(t)
    return filtered
def download_and_extract_fragments(
    url: str, 
    song_id: str, 
    temp_dir: str, 
    times_to_extract: Dict[int, float], 
    duration: float,
    log_prefix: str = "",
    cookies_file: Optional[str] = None
) -> List[dict]:
    """Downloads audio in compressed format and extracts fragments as small WAVs."""
    logger.info(f"   -> {log_prefix} | Downloading full audio...")
    # 1. Download best compressed audio (faster than WAV)
    out_template = os.path.join(temp_dir, f'{song_id}_full.%(ext)s')
    
    node_path = get_node_path()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'noplaylist': True,
        'restrictfilenames': True,
        'sleep_interval': 5,
        'max_sleep_interval': 15,
        'extractor_args': {
            'youtube': { 'player_client': ['web', 'mweb', 'tv'] }
        },
    }
    if node_path:
        ydl_opts['js_runtimes'] = {'node': {'path': node_path}}
    target_cookies = cookies_file or os.path.join('config', 'www.youtube.com_cookies.txt')
    if target_cookies and os.path.exists(target_cookies):
        ydl_opts['cookiefile'] = target_cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            full_audio_path = ydl.prepare_filename(info)

        # 2. Extract fragments using ffmpeg (very fast)
        logger.info(f"   -> {log_prefix} | Extracting {len(times_to_extract)} fragments...")
        fragment_files = []
        for frag_idx, start_sec in times_to_extract.items():
            frag_path = os.path.join(temp_dir, f"{song_id}_f{frag_idx}.wav")
            
            # Use ffmpeg for fast seeking and extraction
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_sec),
                '-t', str(duration),
                '-i', full_audio_path,
                '-ar', '48000',
                '-ac', '1',
                '-vn',
                frag_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            fragment_files.append({
                'index': frag_idx,
                'path': frag_path,
                'start_sec': start_sec
            })

        # Cleanup full compressed file
        if os.path.exists(full_audio_path):
            os.remove(full_audio_path)
            
        return fragment_files
    except Exception as e:
        logger.error(f'   -> {log_prefix} | Download/Extract error: {e}')
        return []
def extract_batch_features(
    fragment_data: List[dict],
    device: torch.device,
    processor: ClapProcessor,
    model: ClapModel,
    log_prefix: str = ""
) -> List[dict]:
    """Processes multiple fragments in a single GPU batch."""
    if not fragment_data:
        return []

    logger.info(f"   -> {log_prefix} | Processing features on {device}...")

    try:
        audio_arrays = []
        valid_fragments = []

        for data in fragment_data:
            path = data['path']
            # Load at 48k for CLAP
            y, _ = librosa.load(path, sr=48_000)
            
            # Sanity checks
            if len(y) < 48_000 * 5 or np.max(np.abs(y)) < 1e-4:
                logger.warning(f'   -> {log_prefix} | Fragment {data["index"]} is too short or silent — skipping.')
                continue
            
            audio_arrays.append(y)
            valid_fragments.append(data)

        if not audio_arrays:
            return []

        # 1. CLAP Batch Inference
        inputs = processor(audio=audio_arrays, sampling_rate=48_000, return_tensors='pt', padding=True).to(device)
        
        if device.type == 'cuda':
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            actual_model = model.module if hasattr(model, 'module') else model
            audio_features = actual_model.get_audio_features(**inputs)
            
            # Handle cases where the model returns a wrapper object instead of a raw tensor
            if not torch.is_tensor(audio_features):
                if hasattr(audio_features, 'pooler_output'):
                    audio_features = audio_features.pooler_output
                elif isinstance(audio_features, (list, tuple)):
                    audio_features = audio_features[0]

            embeddings = audio_features.cpu().float().numpy()

        # 2. Acoustic features (BPM, MFCC) - still sequential as librosa is CPU bound
        results = []
        for i, data in enumerate(valid_fragments):
            y_clap = audio_arrays[i]
            y_features = librosa.resample(y_clap, orig_sr=48_000, target_sr=22_050)
            sr_features = 22_050

            tempo, _ = librosa.beat.beat_track(y=y_features, sr=sr_features)
            mfccs = librosa.feature.mfcc(y=y_features, sr=sr_features, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            bpm = float(np.mean(tempo))
            embedding = embeddings[i]

            record = {
                'fragment_index': data['index'],
                'fragment_start_sec': data['start_sec'],
                'bpm': bpm,
            }
            for j in range(13):
                record[f'mfcc_{j}'] = mfcc_means[j]
            for j in range(512):
                record[f'emb_{j}'] = embedding[j]
            
            results.append(record)

        return results
    except Exception as e:
        logger.error(f'   -> {log_prefix} | Batch processing error: {e}')
        return []

# ── Main Loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latin Music Dataset Generator (Local)")
    parser.add_argument("--csv", type=str, default=os.path.join("data", "spotify_playlists_dataset.csv"), help="Input playlist CSV")
    parser.add_argument("--output", type=str, default=os.path.join("data", "latin_music_features_dataset.csv"), help="Output CSV path")
    parser.add_argument("--temp_dir", type=str, default="temp_audio", help="Temporary directory for audio")
    parser.add_argument("--cookies", type=str, default=os.path.join("config", "www.youtube.com_cookies.txt"), help="Path to YouTube cookies file")
    parser.add_argument("--max_fragments", type=int, default=3, help="Fragments per song")
    parser.add_argument("--frag_duration", type=int, default=30, help="Fragment duration (sec)")
    parser.add_argument("--min_dist", type=int, default=45, help="Min distance between fragments (sec)")
    parser.add_argument("--max_failures", type=int, default=40, help="Abort after N consecutive failures")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of songs to process")
    parser.add_argument("--skip_incomplete", action="store_true", help="Skip songs that already exist in the output CSV but have missing fragments")
    args = parser.parse_args()

    # Hugging Face Login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        logger.warning("HF_TOKEN not found in .env. Model download might be rate-limited.")

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    if device.type == 'cuda':
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    
    MODEL_ID = 'laion/clap-htsat-unfused'
    logger.info(f'Loading CLAP model ({MODEL_ID})...')
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model = ClapModel.from_pretrained(MODEL_ID)
    
    if torch.cuda.device_count() > 1:
        logger.info(f'Enabling DataParallel across {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    # Use half precision for RTX 3060 (significantly faster)
    if device.type == 'cuda':
        model = model.half()
        logger.info("⚡ Using FP16 (Half Precision) for faster GPU inference.")
        
    model.eval()
    logger.info('✅ CLAP model ready.')

    if not check_ffmpeg():
        return

    # Load Data
    if not os.path.exists(args.csv):
        logger.error(f"Input CSV not found: {args.csv}")
        return

    df_input = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df_input)} songs from {args.csv}")
    valid_song_ids = set(df_input['song_id'].tolist())

    # Resume State
    song_state = {}
    if os.path.exists(args.output):
        try:
            df_existing = pd.read_csv(args.output)
            df_existing = df_existing[df_existing['song_id'].isin(valid_song_ids)].copy()
            
            # Sync genre labels
            genre_map = dict(zip(df_input['song_id'], df_input['genre']))
            df_existing['genre'] = df_existing['song_id'].map(genre_map)
            
            for song_id, group in df_existing.groupby('song_id'):
                song_state[song_id] = {
                    'indices': set(group['fragment_index'].tolist()),
                    'times': dict(zip(group['fragment_index'], group['fragment_start_sec'])),
                }
            
            n_complete = sum(1 for s in song_state.values() if len(s['indices']) >= args.max_fragments)
            logger.info(f"Resumed: {n_complete} complete, {len(song_state) - n_complete} incomplete.")
            
            # Write synced data back to output to update genres and remove deleted songs
            df_existing.to_csv(args.output, index=False)
            logger.info(f"Synced genres and cleaned up deleted songs in {args.output}")
        except Exception as e:
            logger.error(f"Error reading existing output: {e}")

    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Main Processing Loop (Sequential)
    pbar = tqdm(total=len(df_input), desc="Processing songs")
    consecutive_failures = 0
    processed_count = 0

    try:
        for idx, row in df_input.iterrows():
            if args.limit and processed_count >= args.limit:
                break

            song_id = row['song_id']
            track = row['track']
            artist = row.get('artist')
            genre = row['genre']

            # Check if song is already complete or should be skipped if incomplete
            if song_id in song_state:
                is_complete = len(song_state[song_id]['indices']) >= args.max_fragments
                if is_complete or args.skip_incomplete:
                    pbar.update(1)
                    continue

            # 1. Search and Download
            state = song_state.get(song_id)
            existing_indices = state['indices'] if state else set()
            target_indices = set(range(args.max_fragments))
            missing_indices = target_indices - existing_indices

            if not missing_indices:
                pbar.update(1)
                continue

            saved_times = state['times'] if state else {}
            log_prefix = f"[{song_id}] {track} - {artist}"
            logger.info(f"\n[Processing] {log_prefix} ({genre})")

            video_info = search_official_video(track, artist, song_id, args.cookies)
            
            if video_info is None:
                logger.warning(f"   -> {log_prefix} | No video found.")
                consecutive_failures += 1
                pbar.update(1)
                if consecutive_failures >= args.max_failures:
                    logger.error(f"🛑 Too many consecutive failures ({consecutive_failures}). Aborting.")
                    break
                continue

            total_duration = float(video_info.get('duration', 0) or 0)
            heatmap = video_info.get('heatmap')
            video_url = video_info.get('webpage_url', '')

            all_times = select_fragment_times(
                heatmap=heatmap,
                total_duration=total_duration,
                n_fragments=args.max_fragments,
                fragment_duration=args.frag_duration,
                min_distance=args.min_dist
            )
            logger.info(f"   -> {log_prefix} | Duration: {total_duration}s | Fragments planned: {len(all_times)}")

            time_map = {idx: saved_times.get(idx, t) for idx, t in enumerate(all_times)}
            times_to_extract = {idx: time_map[idx] for idx in missing_indices if idx in time_map}

            if not times_to_extract:
                logger.warning(f"   -> {log_prefix} | No new fragments to extract (Video might be too short: {total_duration}s)")
                pbar.update(1)
                continue

            fragments = download_and_extract_fragments(
                video_url, song_id, args.temp_dir, times_to_extract, args.frag_duration, log_prefix, args.cookies
            )

            if not fragments:
                logger.warning(f"   -> {log_prefix} | Download failed.")
                consecutive_failures += 1
                pbar.update(1)
                if consecutive_failures >= args.max_failures:
                    logger.error(f"🛑 Too many consecutive failures ({consecutive_failures}). Aborting.")
                    break
                continue

            consecutive_failures = 0

            # 2. Extract Features
            batch_results = extract_batch_features(fragments, device, processor, model, log_prefix)
            
            # 3. Save to CSV
            for features in batch_results:
                row_record = {
                    'genre': genre,
                    'song_id': song_id,
                    **features
                }
                pd.DataFrame([row_record]).to_csv(
                    args.output,
                    mode='a',
                    header=not os.path.exists(args.output),
                    index=False
                )
            
            # 4. Cleanup
            for frag in fragments:
                try: os.remove(frag['path'])
                except: pass
            
            processed_count += 1
            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping...")
    finally:
        pbar.close()

    logger.info("✅ Finished.")

if __name__ == "__main__":
    main()
