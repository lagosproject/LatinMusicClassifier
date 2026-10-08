import os
import re
import shutil
import glob
import subprocess
import time
import logging
import argparse
import unicodedata
import joblib
import torch
import librosa
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
try:
    from music_core import SpotifyClient as MusicCoreSpotifyClient
    HAS_MUSIC_CORE = True
except ImportError:
    HAS_MUSIC_CORE = False

from transformers import ClapModel, ClapProcessor
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm
import yt_dlp
import kagglehub

# ── Configuration & Logging ──────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "subdivide_playlist.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def find_local_audio(track_name: str, artist_name: str, base_path: str = r"B:\Documentos\Musica\FromTidal") -> str:
    """Searches for a matching audio file in the local folder."""
    if not os.path.exists(base_path):
        logger.warning(f"Local audio path not found: {base_path}")
        return None
    
    target_artist = normalize_text(artist_name)
    target_track = normalize_text(track_name)
    
    # Tidal files are often like: "Index - Artist - Title.flac"
    # We'll check all flac and mp3 files in the folder
    files = glob.glob(os.path.join(base_path, "*.flac")) + glob.glob(os.path.join(base_path, "*.mp3"))
    
    for f in files:
        basename = os.path.basename(f)
        norm_basename = normalize_text(basename)
        
        # Check if both artist and track names are in the filename
        if target_artist in norm_basename and target_track in norm_basename:
            return f
            
    return None

def extract_fragments_from_path(audio_path, song_id, temp_dir, n_fragments=3, frag_duration=30):
    """Extracts fragments from a local audio file."""
    try:
        duration = librosa.get_duration(path=audio_path)
        # Uniform fragments (10%, 50%, 80%)
        times = [duration * 0.1, duration * 0.5, duration * 0.8]
        fragment_files = []
        for i, t in enumerate(times):
            frag_path = os.path.join(temp_dir, f"{song_id}_f{i}.wav")
            cmd = ['ffmpeg', '-y', '-ss', str(t), '-t', str(frag_duration), '-i', audio_path, '-ar', '48000', '-ac', '1', '-vn', frag_path]
            subprocess.run(cmd, capture_output=True, check=True)
            fragment_files.append({'path': frag_path, 'start_sec': t})
        return fragment_files, duration
    except Exception as e:
        logger.error(f"Error extracting from {audio_path}: {e}")
        return [], 0

# ── Spotify Setup ────────────────────────────────────────────────────────────
SOURCE_PLAYLIST_ID = "05OTaAnmlHWcj6Qjc5aaIo"
TARGET_PLAYLISTS = {
    'Merengue': '3tqZqFIrDo0EjVofFAE9ns',
    'Salsa': '15t9FgAYOyUHOWx3WjpQg6',
    'Bachata': '3Q42JZmgKO5R31ceKsOQPl',
    'Reggaeton': '7i5VjqTNhc9k4v2utIkpU5' # Mapping Regueton to Reggaeton
}

def get_spotify_client():
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8978/callback")
    scope = "playlist-read-private playlist-modify-private playlist-modify-public"

    if HAS_MUSIC_CORE and client_id and client_secret:
        try:
            logger.info("Initializing Spotify client via MusicCore (rate-limit retry & resilient session)...")
            core_client = MusicCoreSpotifyClient(
                auth_mode="user",
                scope=scope,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
            )
            return core_client.sp
        except Exception as e:
            logger.warning(f"Could not initialize via MusicCore ({e}), falling back to direct spotipy.")

    if not client_id or not client_secret:
        raise ValueError("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set in .env")
        
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
        open_browser=False # Set to False since we are in an agent environment
    )
    return spotipy.Spotify(auth_manager=auth_manager)

# ── Model Loading ────────────────────────────────────────────────────────────
def load_classification_models(model_dir):
    logger.info(f"Loading classification models from {model_dir}...")
    
    if not os.path.exists(os.path.join(model_dir, "scaler.joblib")):
        if os.path.exists(os.path.join(model_dir, "models", "scaler.joblib")):
            model_dir = os.path.join(model_dir, "models")

    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    
    models = {}
    try:
        from lightgbm import LGBMClassifier
        path = os.path.join(model_dir, "lightgbm_model.joblib")
        if os.path.exists(path): models['LightGBM'] = joblib.load(path)
    except Exception as e: logger.warning(f"Failed to load LightGBM: {e}")
    
    try:
        from xgboost import XGBClassifier
        path = os.path.join(model_dir, "xgboost_model.joblib")
        if os.path.exists(path): models['XGBoost'] = joblib.load(path)
    except Exception as e: logger.warning(f"Failed to load XGBoost: {e}")
        
    try:
        from tensorflow import keras
        path = os.path.join(model_dir, "neural_network.keras")
        if os.path.exists(path): models['NeuralNetwork'] = keras.models.load_model(path)
    except Exception as e: logger.warning(f"Failed to load Neural Network: {e}")

    weights = None
    model_order = None
    weights_path = os.path.join(model_dir, "weighted_voting_weights.npy")
    order_path = os.path.join(model_dir, "weighted_voting_model_order.joblib")
    
    if os.path.exists(weights_path) and os.path.exists(order_path):
        weights = np.load(weights_path)
        model_order = joblib.load(order_path)
        logger.info(f"Loaded ensemble weights for: {model_order}")

    return scaler, le, models, weights, model_order

# ── Audio Processing Helpers ─────────────────────────────────────────────────
def get_node_path():
    path = shutil.which("node")
    if not path:
        common_paths = ["C:\\Program Files\\nodejs\\node.exe", "C:\\Program Files (x86)\\nodejs\\node.exe"]
        for p in common_paths:
            if os.path.exists(p): return p
    return path

def _ydl_info_opts(cookies_file: str = None) -> dict:
    node_path = get_node_path()
    opts = {
        'quiet': True, 'no_warnings': True,
        'sleep_interval': 5, 'max_sleep_interval': 15, 'socket_timeout': 30,
        'nocheckcertificate': True, 'format': 'bestaudio/best',
        'extractor_args': {'youtube': {'player_client': ['web', 'mweb', 'tv']}},
    }
    if node_path: opts['js_runtimes'] = {'node': {'path': node_path}}
    target_cookies = cookies_file or os.path.join('config', 'www.youtube.com_cookies.txt')
    if target_cookies and os.path.exists(target_cookies):
        opts['cookiefile'] = target_cookies
    return opts

def search_official_video(query: str, artist: str = None, cookies_file: str = None) -> dict:
    strategies = [
        (f'ytsearch5:{query} - Topic', 'Topic channel'),
        (f'ytsearch5:{query} official audio', 'Official audio'),
        (f'ytsearch5:{query}', 'Plain search'),
    ]
    if artist:
        strategies.extend([
            (f'ytsearch5:{query} {artist} - Topic', 'Topic channel (with artist)'),
            (f'ytsearch5:{query} {artist} official audio', 'Official audio (with artist)'),
            (f'ytsearch5:{query} {artist}', 'Plain search (with artist)'),
        ])

    with yt_dlp.YoutubeDL(_ydl_info_opts(cookies_file)) as ydl:
        for search_query, strategy_name in strategies:
            try:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get('entries', [])
                if not entries: continue
                for entry in entries:
                    uploader = (entry.get('uploader') or "").lower()
                    title = (entry.get('title') or "").lower()
                    if strategy_name == 'Topic channel' and uploader.endswith('- topic'): return entry
                    if not artist or (artist.lower() in uploader or artist.lower() in title): return entry
            except Exception as e: logger.warning(f"Error in strategy {strategy_name}: {e}")
    return None

def download_and_extract_fragments(url, song_id, temp_dir, total_duration, n_fragments=3, frag_duration=30):
    out_template = os.path.join(temp_dir, f'{song_id}_full.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': out_template, 'quiet': True, 'no_warnings': True,
        'nocheckcertificate': True, 'noplaylist': True, 'restrictfilenames': True,
        'extractor_args': {'youtube': {'player_client': ['web', 'mweb', 'tv']}},
    }
    target_cookies = os.path.join('config', 'www.youtube.com_cookies.txt')
    if os.path.exists(target_cookies):
        ydl_opts['cookiefile'] = target_cookies
    node_path = get_node_path()
    if node_path: ydl_opts['js_runtimes'] = {'node': {'path': node_path}}
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            full_audio_path = ydl.prepare_filename(info)

        # Uniform fragments (10%, 50%, 80%)
        times = [total_duration * 0.1, total_duration * 0.5, total_duration * 0.8]
        fragment_files = []
        for i, t in enumerate(times):
            frag_path = os.path.join(temp_dir, f"{song_id}_f{i}.wav")
            cmd = ['ffmpeg', '-y', '-ss', str(t), '-t', str(frag_duration), '-i', full_audio_path, '-ar', '48000', '-ac', '1', '-vn', frag_path]
            subprocess.run(cmd, capture_output=True, check=True)
            fragment_files.append({'path': frag_path, 'start_sec': t})
        
        if os.path.exists(full_audio_path): os.remove(full_audio_path)
        return fragment_files
    except Exception as e:
        logger.error(f"Download/Extract error for {song_id}: {e}")
        return []

def extract_features(fragment_data, device, processor, model):
    if not fragment_data: return []
    try:
        audio_arrays = []
        for data in fragment_data:
            y, _ = librosa.load(data['path'], sr=48_000)
            audio_arrays.append(y)
            
        inputs = processor(audio=audio_arrays, sampling_rate=48_000, return_tensors='pt', padding=True).to(device)
        if device.type == 'cuda': inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            output = model.get_audio_features(**inputs)
            if not torch.is_tensor(output):
                output = output.pooler_output if hasattr(output, 'pooler_output') else output[0]
            embeddings = output.cpu().float().numpy()

        results = []
        for i, data in enumerate(fragment_data):
            y_clap = audio_arrays[i]
            y_features = librosa.resample(y_clap, orig_sr=48_000, target_sr=22_050)
            tempo, _ = librosa.beat.beat_track(y=y_features, sr=22_050)
            mfccs = np.mean(librosa.feature.mfcc(y=y_features, sr=22_050, n_mfcc=13), axis=1)
            
            record = {'bpm': float(np.mean(tempo)), 'fragment_start_sec': data['start_sec']}
            for j in range(13): record[f'mfcc_{j}'] = mfccs[j]
            for j in range(512): record[f'emb_{j}'] = embeddings[i][j]
            results.append(record)
        return results
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return []

# ── Inference ────────────────────────────────────────────────────────────────
def classify_song(features, scaler, le, models, weights, model_order):
    X = pd.DataFrame(features).drop(columns=['fragment_start_sec'])
    X_scaled = scaler.transform(X)
    
    all_probs = {}
    for name, model in models.items():
        if name == 'NeuralNetwork': all_probs[name] = model.predict(X_scaled, verbose=0)
        else: all_probs[name] = model.predict_proba(X_scaled)
    
    if weights is not None and all(m in all_probs for m in model_order):
        final_probs = np.zeros_like(all_probs[model_order[0]])
        for w, name in zip(weights, model_order): final_probs += w * all_probs[name]
    else:
        final_probs = np.mean(list(all_probs.values()), axis=0)
    
    # Mean-pooling across fragments
    mean_probs = np.mean(final_probs, axis=0)
    predicted_idx = np.argmax(mean_probs)
    return le.classes_[predicted_idx], mean_probs[predicted_idx]

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0, help="Skip first N tracks")
    parser.add_argument("--local_path", type=str, default=r"B:\Documentos\Musica\FromTidal")
    parser.add_argument("--no_youtube", action="store_true", help="Disable YouTube fallback")
    args = parser.parse_args()

    # 1. Init Spotify
    sp = get_spotify_client()
    user = sp.current_user()
    logger.info(f"Authenticated as: {user['display_name']}")

    # 2. Init Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    MODEL_ID = 'laion/clap-htsat-unfused'
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    clap_model = ClapModel.from_pretrained(MODEL_ID).to(device)
    if device.type == 'cuda': clap_model = clap_model.half()
    clap_model.eval()

    model_path = kagglehub.model_download("g5fighter/latin-music-genre-classifiers/tensorFlow2/latin-music-genre-classifier")
    scaler, le, cls_models, weights, model_order = load_classification_models(model_path)

    # 3. Load Datasets for Lookup
    ground_truth_file = os.path.join("data", "spotify_playlists_dataset.csv")
    ground_truth_map = {}
    if os.path.exists(ground_truth_file):
        gt_df = pd.read_csv(ground_truth_file)
        ground_truth_map = dict(zip(gt_df['song_id'], gt_df['genre']))
        logger.info(f"Loaded {len(ground_truth_map)} ground truth labels.")

    features_db_file = os.path.join("data", "latin_music_features_dataset.csv")
    existing_features_ids = set()
    if os.path.exists(features_db_file):
        existing_features_ids = set(pd.read_csv(features_db_file, usecols=['song_id'])['song_id'].unique())
        logger.info(f"Found {len(existing_features_ids)} songs already in features dataset.")

    new_features_file = os.path.join("data", "subdivided_playlist_features.csv")

    # 4. Load Progress
    progress_file = os.path.join("logs", "classification_progress.csv")
    if os.path.exists(progress_file): progress_df = pd.read_csv(progress_file)
    else: progress_df = pd.DataFrame(columns=['song_id', 'track', 'artist', 'predicted_genre', 'confidence'])

    # 5. Fetch Source Playlist
    logger.info(f"Fetching tracks from source playlist: {SOURCE_PLAYLIST_ID}")
    results = sp.playlist_items(SOURCE_PLAYLIST_ID)
    tracks = []
    while results:
        for entry in results['items']:
            if not entry:
                continue
            
            # Match robust logic from notebook: check 'item' then 'track'
            t_obj = entry.get('item') or entry.get('track')
            
            # Robust validation
            if not t_obj or not isinstance(t_obj, dict):
                continue
            if t_obj.get('type') == 'episode': # skip podcasts
                continue
            artists = t_obj.get('artists')
            if not (artists and isinstance(artists, list) and len(artists) > 0):
                continue
            if not t_obj.get('id'):
                continue

            artist_name = t_obj['artists'][0].get('name', 'Unknown Artist')
            tracks.append({
                'id': t_obj['id'],
                'uri': t_obj['uri'],
                'name': t_obj.get('name', "Unknown Track"),
                'artist': artist_name
            })
        results = sp.next(results) if results['next'] else None
    
    if args.skip: tracks = tracks[args.skip:]
    if args.limit: tracks = tracks[:args.limit]
    logger.info(f"Processing {len(tracks)} tracks (Skipped: {args.skip}, Limit: {args.limit or 'None'}).")

    # 6. Cache Target Playlists to prevent duplicates
    logger.info("Caching target playlist items to prevent duplicates...")
    playlist_item_cache = {} # {playlist_id: set(track_ids)}
    for g, pid in TARGET_PLAYLISTS.items():
        if not pid: continue
        try:
            p_items = []
            res = sp.playlist_items(pid, fields="items(track(id)),next")
            while res:
                p_items.extend([it['track']['id'] for it in res['items'] if it.get('track')])
                res = sp.next(res) if res['next'] else None
            playlist_item_cache[pid] = set(p_items)
            logger.info(f"  - Cached {len(p_items)} items for {g}")
        except Exception as e:
            logger.warning(f"  - Could not cache {g}: {e}")
            playlist_item_cache[pid] = set()

    # 7. Process Tracks
    temp_dir = "temp_audio_subdivide"
    os.makedirs(temp_dir, exist_ok=True)

    for track in tqdm(tracks, desc="Processing tracks"):
        song_id = track['id']
        if song_id in progress_df['song_id'].values:
            continue

        genre = None
        confidence = 1.0
        features = None

        # Tier 1: Check Ground Truth
        if song_id in ground_truth_map:
            genre = ground_truth_map[song_id]
            logger.info(f"Found ground truth for '{track['name']}': {genre}")

        # Tier 2: Check if features needed
        features_needed = song_id not in existing_features_ids
        if os.path.exists(new_features_file):
            try:
                processed_in_session = song_id in pd.read_csv(new_features_file, usecols=['song_id'])['song_id'].values
                if processed_in_session: features_needed = False
            except: pass

        if features_needed or genre is None:
            # Need to get audio
            audio_path = find_local_audio(track['name'], track['artist'], args.local_path)
            fragment_files = []
            
            if audio_path:
                logger.info(f"Local audio found for '{track['name']}': {audio_path}")
                fragment_files, _ = extract_fragments_from_path(audio_path, song_id, temp_dir)
            elif not args.no_youtube:
                logger.info(f"Searching YouTube for '{track['name']}' by '{track['artist']}'...")
                query = f"{track['name']} {track['artist']}"
                video = search_official_video(query, track['artist'])
                if video:
                    duration = video.get('duration') or 240
                    fragment_files = download_and_extract_fragments(video['webpage_url'], song_id, temp_dir, duration)
                else:
                    logger.warning(f"Could not find audio on YouTube for {track['name']}")
                    continue
            else:
                logger.info(f"Skipping YouTube search for '{track['name']}' (--no_youtube)")
                # If we have ground truth, we can still add to Spotify, but we can't extract features.
                if genre is None:
                    continue
            
            if fragment_files:
                features = extract_features(fragment_files, device, processor, clap_model)
                
                # Save features if needed
                if features_needed and features:
                    feat_df = pd.DataFrame(features)
                    feat_df.insert(0, 'fragment_index', range(len(features)))
                    feat_df.insert(0, 'song_id', song_id)
                    feat_df.insert(0, 'genre', genre if genre else "Unknown")
                    feat_df.to_csv(new_features_file, mode='a', header=not os.path.exists(new_features_file), index=False)
                    logger.info(f"Saved {len(features)} fragments to {new_features_file}")

                for f in fragment_files:
                    if os.path.exists(f['path']): os.remove(f['path'])

        # Final classification if not from ground truth
        if genre is None:
            if features:
                genre, confidence = classify_song(features, scaler, le, cls_models, weights, model_order)
                logger.info(f"Classified '{track['name']}' as {genre} (conf: {confidence:.2f})")
            else:
                logger.warning(f"Skipping '{track['name']}' - no features extracted.")
                continue

        # Add to Spotify
        target_playlist_id = TARGET_PLAYLISTS.get(genre)
        if target_playlist_id:
            # Check for duplicates in cache
            if song_id in playlist_item_cache.get(target_playlist_id, set()):
                logger.info(f"'{track['name']}' already in {genre} playlist (cached).")
            else:
                try:
                    sp.playlist_add_items(target_playlist_id, [track['uri']])
                    logger.info(f"Added '{track['name']}' to {genre} playlist.")
                    # Update cache
                    playlist_item_cache[target_playlist_id].add(song_id)
                except Exception as e:
                    logger.error(f"Error adding to Spotify: {e}")
        else:
            logger.warning(f"No target playlist for genre: {genre}")

        # Save Progress
        new_row = pd.DataFrame([{
            'song_id': song_id,
            'track': track['name'],
            'artist': track['artist'],
            'predicted_genre': genre,
            'confidence': confidence
        }])
        new_row.to_csv(progress_file, mode='a', header=not os.path.exists(progress_file), index=False)
        progress_df = pd.concat([progress_df, new_row], ignore_index=True)

    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Playlist subdivision complete.")

if __name__ == "__main__":
    main()
