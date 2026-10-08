import os
import joblib
import pandas as pd
import numpy as np
import kagglehub
import argparse
from tqdm import tqdm

# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_models(model_dir):
    """Loads all models and preprocessing artifacts from the specified directory."""
    print(f"Searching for model assets under {model_dir}...")
    
    # Recursively find directory containing scaler.joblib
    found_dir = None
    if os.path.exists(os.path.join(model_dir, "scaler.joblib")):
        found_dir = model_dir
    else:
        for root, dirs, files in os.walk(model_dir):
            if "scaler.joblib" in files:
                found_dir = root
                break
                
    if found_dir:
        model_dir = found_dir
        print(f"Using model directory: {model_dir}")
    else:
        print(f"Warning: Could not find scaler.joblib in {model_dir} or subdirectories.")

    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    
    # Load manifest if available
    manifest = None
    manifest_path = os.path.join(model_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            import json
            with open(manifest_path, 'r') as fh:
                manifest = json.load(fh)
            print(f"Loaded manifest.json: {manifest}")
        except Exception as e:
            print(f"Warning: Failed to load manifest.json: {e}")

    # Load base models
    models = {}
    
    try:
        from lightgbm import LGBMClassifier
        path = os.path.join(model_dir, "lightgbm_model.joblib")
        if os.path.exists(path):
            models['LightGBM'] = joblib.load(path)
    except Exception as e:
        print(f"Warning: Failed to load LightGBM: {e}")
    
    try:
        from xgboost import XGBClassifier
        path = os.path.join(model_dir, "xgboost_model.joblib")
        if os.path.exists(path):
            models['XGBoost'] = joblib.load(path)
    except Exception as e:
        print(f"Warning: Failed to load XGBoost: {e}")
        
    try:
        import tensorflow as tf
        from tensorflow import keras
        path = os.path.join(model_dir, "neural_network.keras")
        if os.path.exists(path):
            models['NeuralNetwork'] = keras.models.load_model(path)
    except Exception as e:
        print(f"Warning: Failed to load Neural Network: {e}")

    # Load ensemble weights
    weights = None
    model_order = None
    weights_path = os.path.join(model_dir, "weighted_voting_weights.npy")
    order_path = os.path.join(model_dir, "weighted_voting_model_order.joblib")
    
    if os.path.exists(weights_path) and os.path.exists(order_path):
        weights = np.load(weights_path)
        model_order = joblib.load(order_path)
        print(f"Loaded ensemble weights for: {model_order}")

    # Load stacking meta-model
    stacking_meta = None
    stacking_path = os.path.join(model_dir, "stacking_meta.joblib")
    if os.path.exists(stacking_path):
        try:
            stacking_meta = joblib.load(stacking_path)
            print("Loaded Stacking meta-model.")
        except Exception as e:
            print(f"Warning: Failed to load Stacking meta-model: {e}")

    return scaler, le, models, weights, model_order, stacking_meta, manifest

def preprocess_data(df, scaler):
    """Cleans and scales the dataset according to the rules in the training notebook."""
    print(f"Original shape: {df.shape}")
    
    # 1. Drop songs with late fragments (>= 300s)
    invalid_songs = df[df['fragment_start_sec'] >= 300]['song_id'].unique()
    df_clean = df[~df['song_id'].isin(invalid_songs)].copy()
    print(f"After removing late-fragment songs: {df_clean.shape} (dropped {len(invalid_songs)} songs)")
    
    # 2. Filter BPM outliers (octave errors)
    df_clean['median_bpm'] = df_clean.groupby('song_id')['bpm'].transform('median')
    tol = 0.08
    is_normal = (df_clean['bpm'] - df_clean['median_bpm']).abs() / df_clean['median_bpm'] <= tol
    is_double  = (df_clean['bpm'] - df_clean['median_bpm'] * 2).abs() / (df_clean['median_bpm'] * 2) <= tol
    is_half    = (df_clean['bpm'] - df_clean['median_bpm'] / 2).abs() / (df_clean['median_bpm'] / 2) <= tol
    
    df_filtered = df_clean[is_normal | is_double | is_half].drop(columns=['median_bpm']).copy()
    print(f"After BPM outlier removal: {df_filtered.shape}")
    
    # 3. Extract features
    cols_to_drop = ['genre', 'song_id', 'fragment_index', 'fragment_start_sec', 'track']
    X = df_filtered.drop(columns=cols_to_drop, errors='ignore')
    
    # 4. Scale features
    X_scaled = scaler.transform(X)
    
    return df_filtered, X_scaled

def main():
    parser = argparse.ArgumentParser(description="Predict genres and identify potential mislabels.")
    parser.add_argument("--csv", type=str, default=os.path.join("data", "latin_music_features_dataset.csv"), help="Dataset CSV path")
    parser.add_argument("--metadata", type=str, default=os.path.join("data", "spotify_playlists_dataset.csv"), help="Metadata CSV path (to get artist/track names)")
    parser.add_argument("--output", type=str, default=os.path.join("out", "potential_mislabels.csv"), help="Output report CSV path")
    parser.add_argument("--confidence_threshold", type=float, default=0.85, help="Confidence threshold to filter low-confidence correct labels (mislabels are now always reported)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of songs for testing")
    parser.add_argument("--force_download", action="store_true", help="Force redownload of the model from Kaggle")
    parser.add_argument("--ensemble", type=str, choices=["auto", "stacking", "weighted_voting", "average"], default="auto",
                        help="Ensemble method to use (default: auto, which resolves to stacking if meta-model is available, then weighted_voting, then average)")
    args = parser.parse_args()

    # 1. Download/Path to models
    try:
        model_path = kagglehub.model_download(
            "g5fighter/latin-music-genre-classifiers/tensorFlow2/latin-music-genre-classifier",
            force_download=args.force_download
        )
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # 2. Load Models
    scaler, le, models, ensemble_weights, model_order, stacking_meta, manifest = load_models(model_path)
    
    if not models:
        print("No models loaded. Please ensure lightgbm, xgboost, or tensorflow are installed.")
        return

    # Determine the pooling strategy to use from manifest (default 'mean')
    pooling = 'mean'
    if manifest and 'best_pooling' in manifest:
        pooling = manifest['best_pooling']
        print(f"Manifest specified pooling strategy: '{pooling}'")
    else:
        print(f"No pooling strategy specified in manifest, defaulting to '{pooling}'")

    # 3. Load Data
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        return
        
    df = pd.read_csv(args.csv)
    
    if args.limit:
        song_ids = df['song_id'].unique()[:args.limit]
        df = df[df['song_id'].isin(song_ids)].copy()
        print(f"Limiting to {args.limit} songs...")

    # 4. Preprocess
    df_filtered, X_scaled = preprocess_data(df, scaler)
    
    # 5. Inference & Song-Level Aggregation
    prob_cols = le.classes_.tolist()
    song_probs_by_model = {}
    unique_song_ids = df_filtered['song_id'].unique()
    
    for name, model in models.items():
        print(f"Running inference with {name}...")
        if name == 'NeuralNetwork':
            preds = model.predict(X_scaled, verbose=0)
        else:
            preds = model.predict_proba(X_scaled)
            
        df_model_p = pd.DataFrame(preds, columns=prob_cols)
        df_model_p['song_id'] = df_filtered['song_id'].values
        
        # Aggregate fragment probabilities to song level
        if pooling == 'max':
            song_probs_model = df_model_p.groupby('song_id')[prob_cols].max()
        else:
            song_probs_model = df_model_p.groupby('song_id')[prob_cols].mean()
            
        # Reindex to ensure consistent order of songs across all models
        song_probs_model = song_probs_model.reindex(unique_song_ids)
        song_probs_by_model[name] = song_probs_model

    # 6. Ensemble Combination
    ensemble_method = args.ensemble
    if ensemble_method == "auto":
        if stacking_meta is not None:
            ensemble_method = "stacking"
        elif ensemble_weights is not None:
            ensemble_method = "weighted_voting"
        else:
            ensemble_method = "average"
            
    print(f"Using ensemble method: {ensemble_method}")
    
    if ensemble_method == "stacking":
        if stacking_meta is None:
            print("Error: Stacking meta-model (stacking_meta.joblib) not found, falling back to average.")
            ensemble_method = "average"
        elif model_order is None or not all(m in song_probs_by_model for m in model_order):
            print("Error: Missing base models or model order for stacking, falling back to average.")
            ensemble_method = "average"
            
    if ensemble_method == "stacking":
        print("Applying stacking ensemble...")
        # Concatenate the song-level probabilities of base models in model_order column-wise
        stack_parts = [song_probs_by_model[name].values for name in model_order]
        X_stack = np.hstack(stack_parts)
        
        # Predict class probabilities using stacking meta-model
        final_probs = stacking_meta.predict_proba(X_stack)
        song_agg = pd.DataFrame(final_probs, columns=prob_cols, index=unique_song_ids)
        
    elif ensemble_method == "weighted_voting":
        if ensemble_weights is None or model_order is None or not all(m in song_probs_by_model for m in model_order):
            print("Warning: Missing weights or base models for weighted voting, falling back to average.")
            ensemble_method = "average"
        else:
            print("Applying weighted ensemble...")
            final_probs = np.zeros((len(unique_song_ids), len(prob_cols)))
            for weight, name in zip(ensemble_weights, model_order):
                final_probs += weight * song_probs_by_model[name].values
            song_agg = pd.DataFrame(final_probs, columns=prob_cols, index=unique_song_ids)
            
    if ensemble_method == "average":
        print("Averaging available models...")
        final_probs = np.mean([song_probs_by_model[name].values for name in song_probs_by_model], axis=0)
        song_agg = pd.DataFrame(final_probs, columns=prob_cols, index=unique_song_ids)

    # 7. Get original genre aligned with unique_song_ids
    original_genres = df_filtered.groupby('song_id')['genre'].first().reindex(unique_song_ids)
    
    # 8. Identify Mislabels
    predictions = song_agg.idxmax(axis=1)
    confidences = song_agg.max(axis=1)
    
    results = pd.DataFrame({
        'song_id': unique_song_ids,
        'original_genre': original_genres.values,
        'predicted_genre': predictions.values,
        'confidence': confidences.values
    })
    
    # Merge with metadata (artist, track)
    if os.path.exists(args.metadata):
        print(f"Enriching results with metadata from {args.metadata}...")
        df_meta = pd.read_csv(args.metadata)
        # Keep only unique song_id metadata
        df_meta = df_meta[['song_id', 'artist', 'track']].drop_duplicates(subset=['song_id'])
        results = results.merge(df_meta, on='song_id', how='left')
        
        # Reorder columns for better readability
        cols = ['song_id', 'artist', 'track', 'original_genre', 'predicted_genre', 'confidence']
        results = results[cols]
    else:
        print(f"Warning: Metadata file not found at {args.metadata}. Results will only contain song_id.")

    results['is_mismatch'] = results['original_genre'] != results['predicted_genre']
    
    # Flag ALL mismatches as mislabels (ignoring confidence threshold)
    mislabels = results[results['is_mismatch']].copy()
    mislabels = mislabels.sort_values(by='confidence', ascending=False)
    
    # Flag low confidence correct labels (model got it right but is unsure)
    low_conf_correct = results[(~results['is_mismatch']) & (results['confidence'] < args.confidence_threshold)].copy()
    low_conf_correct = low_conf_correct.sort_values(by='confidence', ascending=True)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"\nFound {len(mislabels)} potential mislabels out of {len(results)} total songs.")
    print(f"Found {len(low_conf_correct)} low-confidence correct labels.")
    
    # Save mislabels
    mislabels.to_csv(args.output, index=False)
    print(f"Mislabels report saved to {args.output}")
    
    # Save low confidence correct
    low_conf_path = args.output.replace("potential_mislabels.csv", "low_confidence_correct.csv")
    if low_conf_path == args.output: # Fallback if user used a different filename
        low_conf_path = os.path.join(output_dir, "low_confidence_correct.csv")
    
    low_conf_correct.to_csv(low_conf_path, index=False)
    print(f"Low-confidence correct labels saved to {low_conf_path}")
    
    # Also save the full results for complete analysis
    full_output = args.output.replace(".csv", "_full.csv")
    results.to_csv(full_output, index=False)
    print(f"Full prediction report saved to {full_output}")

if __name__ == "__main__":
    main()
