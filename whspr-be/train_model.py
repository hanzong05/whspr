"""
Train Emotion Recognition Model
Supports training with labeled audio datasets
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import librosa
import warnings

from mfcc_feature_extraction import MFCCFeatureExtractor
from ml_classifier import EmotionClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

EMOTION_MAPPING = {
    # RAVDESS dataset mapping
    '01': 'neutral',
    '02': 'neutral',       # calm -> neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'sad',           # fearful -> sad (was frustrated)
    '07': 'angry',         # disgust -> angry (was frustrated)
    '08': 'satisfied',     # surprised -> satisfied

    # TESS (Toronto Emotional Speech Set) mapping
    'angry': 'angry',
    'disgust': 'angry',        # was frustrated, now angry
    'fear': 'sad',             # was frustrated, now sad
    'happy': 'happy',
    'neutral': 'neutral',
    'pleasant_surprise': 'satisfied',
    'sad': 'sad',

    # CREMA-D mapping
    'ANG': 'angry',
    'DIS': 'angry',        # was frustrated, now angry
    'FEA': 'sad',          # was frustrated, now sad
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',

    # Direct mapping
    'frustrated': 'frustrated',
    'satisfied': 'satisfied',
}

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset_from_folder(audio_folder, label_mapping=None):
    """
    Load audio files from a folder structure like:
    audio_folder/
        angry/
            file1.wav
        happy/
            file1.wav
        etc.
    """
    print(f"\n{'='*70}")
    print(f"Loading Dataset from Folder: {audio_folder}")
    print(f"{'='*70}\n")

    audio_folder = Path(audio_folder)
    dataset = []

    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

    for emotion_folder in audio_folder.iterdir():
        if emotion_folder.is_dir():
            raw_name = emotion_folder.name.strip()
            lower_name = raw_name.lower()
            upper_name = raw_name.upper()

            if label_mapping is None:
                label_mapping = EMOTION_MAPPING

            if raw_name in label_mapping:
                emotion_label = label_mapping[raw_name]
            elif lower_name in label_mapping:
                emotion_label = label_mapping[lower_name]
            elif upper_name in label_mapping:
                emotion_label = label_mapping[upper_name]
            else:
                emotion_label = lower_name

            print(f"📁 Scanning {raw_name} folder → Label: {emotion_label}")

            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(emotion_folder.glob(f'*{ext}'))

            for audio_file in audio_files:
                dataset.append((str(audio_file), emotion_label))

            print(f"   Found {len(audio_files)} files")

    print(f"\n✅ Total files loaded: {len(dataset)}")
    print(f"📊 Emotion distribution:")

    emotion_counts = {}
    for _, emotion in dataset:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count}")

    return dataset


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataset(dataset, feature_extractor):
    """
    Extract MFCC features from all audio files in dataset.
    Returns (features_array, labels_array).
    """
    print(f"\n{'='*70}")
    print(f"Extracting MFCC Features")
    print(f"{'='*70}\n")

    features_list = []
    labels_list = []
    failed_count = 0

    for i, (audio_path, label) in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(dataset)} files processed...")

        try:
            features = feature_extractor.extract_features(audio_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
            else:
                failed_count += 1
        except Exception as e:
            print(f"⚠️  Error processing {audio_path}: {e}")
            failed_count += 1
            continue

    print(f"\n✅ Feature extraction complete!")
    print(f"   Successful: {len(features_list)}")
    print(f"   Failed: {failed_count}")

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\n📊 Feature matrix shape: {X.shape}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features per sample: {X.shape[1]}")

    return X, y


# ============================================================================
# BALANCING
# ============================================================================

def balance_dataset(X, y):
    """
    Balance classes by:
    - Undersampling classes above the median count
    - Oversampling classes below the median count
    Returns balanced (X, y).
    """
    print(f"\n{'='*70}")
    print("Balancing Dataset")
    print(f"{'='*70}\n")

    print("📊 Class distribution BEFORE balancing:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"   {emotion}: {count}")

    indices_by_class = defaultdict(list)
    for i, label in enumerate(y):
        indices_by_class[label].append(i)

    # Use median as target so we don't lose too much data from majority classes
    target_count = int(np.median([len(v) for v in indices_by_class.values()]))
    print(f"\n🎯 Target samples per class: {target_count}")

    balanced_indices = []
    for label, indices in indices_by_class.items():
        if len(indices) >= target_count:
            # Undersample
            chosen = np.random.choice(indices, target_count, replace=False)
        else:
            # Oversample minority class
            chosen = np.random.choice(indices, target_count, replace=True)
        balanced_indices.extend(chosen)

    np.random.shuffle(balanced_indices)
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    print("\n📊 Class distribution AFTER balancing:")
    unique2, counts2 = np.unique(y_balanced, return_counts=True)
    for emotion, count in zip(unique2, counts2):
        print(f"   {emotion}: {count}")

    return X_balanced, y_balanced


# ============================================================================
# TRAINING
# ============================================================================

def train_and_save_model(X, y, classifier_type='svm', save_path='models/'):
    """
    Balance data, train emotion classifier, and save it.
    Returns (classifier, training_history).
    """
    print(f"\n{'='*70}")
    print(f"Training {classifier_type.upper()} Classifier")
    print(f"{'='*70}\n")

    # Balance before training
    X, y = balance_dataset(X, y)

    # Create and train classifier
    classifier = EmotionClassifier(classifier_type=classifier_type)

    classifier.train(
        X, y,
        test_size=0.2,
        C=10,           # was 50 — reduced to prevent overfitting to majority class
        gamma='scale',  # was 0.01 — let sklearn auto-tune based on feature variance
        kernel='rbf'
    )

    # Save model
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / f'{classifier_type}_emotion_model.pkl'

    # Backup existing model if present
    if model_file.exists():
        backup_file = save_path / f'{classifier_type}_emotion_model_backup.pkl'
        import shutil
        shutil.copy(str(model_file), str(backup_file))
        print(f"📦 Backup saved to: {backup_file}")

    classifier.save_model(str(model_file))
    print(f"\n✅ Model saved to: {model_file}")

    history = getattr(classifier, 'training_history', {})
    return classifier, history


# ============================================================================
# SAMPLE DATA GENERATOR (for testing only)
# ============================================================================

def create_sample_training_data():
    """
    Create a small synthetic dataset for testing the pipeline.
    NOT suitable for real emotion recognition.
    """
    print(f"\n{'='*70}")
    print("Creating Sample Training Data")
    print(f"{'='*70}\n")

    import soundfile as sf

    sample_folder = Path("sample_training_data")
    emotions = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'satisfied']

    for emotion in emotions:
        emotion_folder = sample_folder / emotion
        emotion_folder.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            duration = 3
            sr = 22050

            if emotion == 'angry':
                freq = np.random.randint(300, 400)
            elif emotion == 'happy':
                freq = np.random.randint(400, 500)
            elif emotion == 'sad':
                freq = np.random.randint(150, 250)
            elif emotion == 'frustrated':
                freq = np.random.randint(280, 380)
            elif emotion == 'satisfied':
                freq = np.random.randint(350, 450)
            else:
                freq = np.random.randint(250, 350)

            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            audio += 0.05 * np.random.randn(len(audio))

            filename = emotion_folder / f"{emotion}_sample_{i+1}.wav"
            sf.write(filename, audio, sr)

        print(f"✅ Created {emotion} samples")

    print(f"\n📁 Sample data created in: {sample_folder}")
    print(f"⚠️  NOTE: This is synthetic data for TESTING ONLY!")
    print(f"   For real emotion recognition, use actual emotional speech datasets.")

    return str(sample_folder)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print("CSR EMOTION RECOGNITION - MODEL TRAINING")
    print(f"{'='*70}\n")

    print("Options:")
    print("1. Create sample training data (for testing)")
    print("2. Train from existing dataset folder")
    print("3. Exit")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == '1':
        data_folder = create_sample_training_data()
        print("\nNow training on sample data...")

    elif choice == '2':
        print("\nSupported datasets: RAVDESS, TESS, CREMA-D, or any custom folder structure")
        print("Folder should contain subfolders named with emotion labels")
        print("(e.g., angry/, happy/, sad/, neutral/, frustrated/, satisfied/)\n")
        data_folder = input("Enter path to your dataset folder: ").strip()

        if not os.path.exists(data_folder):
            print(f"❌ Error: Folder '{data_folder}' not found!")
            return

    elif choice == '3':
        print("Exiting...")
        return

    else:
        print("Invalid choice!")
        return

    # Load dataset
    dataset = load_dataset_from_folder(data_folder)

    if len(dataset) == 0:
        print("❌ No audio files found! Please check your dataset structure.")
        return

    # Initialize feature extractor
    print("\nInitializing MFCC Feature Extractor...")
    feature_extractor = MFCCFeatureExtractor()

    # Extract features
    X, y = extract_features_from_dataset(dataset, feature_extractor)

    if len(X) == 0:
        print("❌ No features extracted! Check your audio files.")
        return

    # Choose classifier
    print("\nWhich classifier would you like to train?")
    print("1. SVM (Support Vector Machine) - Recommended")
    print("2. Random Forest")
    print("3. KNN (K-Nearest Neighbors)")
    print("4. All three")

    clf_choice = input("\nEnter choice (1/2/3/4): ").strip()

    classifiers_to_train = []
    if clf_choice == '1':
        classifiers_to_train = ['svm']
    elif clf_choice == '2':
        classifiers_to_train = ['rf']
    elif clf_choice == '3':
        classifiers_to_train = ['knn']
    elif clf_choice == '4':
        classifiers_to_train = ['svm', 'rf', 'knn']
    else:
        print("Invalid choice, training SVM by default")
        classifiers_to_train = ['svm']

    # Train
    results_summary = []
    for clf_type in classifiers_to_train:
        classifier, history = train_and_save_model(X, y, classifier_type=clf_type)
        accuracy = history.get('test_accuracy', 0) if isinstance(history, dict) else 0
        results_summary.append({
            'type': clf_type,
            'accuracy': accuracy,
            'training_samples': len(X),
        })

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")

    for result in results_summary:
        print(
            f"{result['type'].upper()}: Accuracy = {result['accuracy']:.2%} "
            f"(trained on {result['training_samples']} samples)"
        )

    print(f"\n✅ Models saved in: models/")
    print(f"\n🔄 Restart your backend server to load the new models!")


# ============================================================================
# WRAPPER FOR EXTERNAL USE
# ============================================================================

def train_from_labeled_recordings(data_folder, classifiers_to_train=None):
    """
    Train emotion recognition models from labeled dataset folder.
    Used by setup scripts.
    """
    print(f"\n{'='*70}")
    print("TRAINING EMOTION RECOGNITION MODELS")
    print(f"{'='*70}\n")

    if classifiers_to_train is None:
        classifiers_to_train = ['svm']

    if not os.path.exists(data_folder):
        print(f"❌ Error: Folder '{data_folder}' not found!")
        return None

    dataset = load_dataset_from_folder(data_folder)

    if len(dataset) == 0:
        print("❌ No audio files found!")
        return None

    print("\nInitializing MFCC Feature Extractor...")
    feature_extractor = MFCCFeatureExtractor()

    X, y = extract_features_from_dataset(dataset, feature_extractor)

    if len(X) == 0:
        print("❌ No features extracted!")
        return None

    results_summary = []
    for clf_type in classifiers_to_train:
        classifier, history = train_and_save_model(X, y, classifier_type=clf_type)
        accuracy = history.get('test_accuracy', 0) if isinstance(history, dict) else 0
        results_summary.append({
            'type': clf_type,
            'accuracy': accuracy,
            'training_samples': len(X),
        })

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")

    for result in results_summary:
        print(
            f"{result['type'].upper()}: Accuracy = {result['accuracy']:.2%} "
            f"(trained on {result['training_samples']} samples)"
        )

    print(f"\n✅ Models saved in: models/")
    print(f"\n🔄 Restart your backend server to load the new models!")

    return results_summary


if __name__ == "__main__":
    main()