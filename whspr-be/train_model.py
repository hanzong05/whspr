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
import librosa
import warnings

from mfcc_feature_extraction import MFCCFeatureExtractor
from ml_classifier import EmotionClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Map your dataset emotion labels to the 6 emotions used in the system
EMOTION_MAPPING = {
    # RAVDESS dataset mapping (matches setup_ravdess.py)
    '01': 'neutral',
    '02': 'neutral',  # calm -> neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'frustrated',  # fearful -> frustrated
    '07': 'frustrated',  # disgust -> frustrated
    '08': 'satisfied',  # surprised -> satisfied

    # Or direct mapping for simpler datasets
    'angry': 'angry',
    'happy': 'happy',
    'sad': 'sad',
    'neutral': 'neutral',
    'frustrated': 'frustrated',
    'satisfied': 'satisfied',
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_dataset_from_folder(audio_folder, label_mapping=None):
    """
    Load audio files from a folder structure like:
    audio_folder/
        angry/
            file1.wav
            file2.wav
        happy/
            file1.wav
            file2.wav
        etc.

    Args:
        audio_folder (str): Path to folder containing emotion subfolders
        label_mapping (dict): Optional mapping to rename labels

    Returns:
        list: List of (audio_path, emotion_label) tuples
    """
    print(f"\n{'='*70}")
    print(f"Loading Dataset from Folder: {audio_folder}")
    print(f"{'='*70}\n")

    audio_folder = Path(audio_folder)
    dataset = []

    # Supported audio formats
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

    # Scan for emotion folders
    for emotion_folder in audio_folder.iterdir():
        if emotion_folder.is_dir():
            emotion_name = emotion_folder.name.lower()

            # Apply mapping if provided
            if label_mapping and emotion_name in label_mapping:
                emotion_label = label_mapping[emotion_name]
            else:
                emotion_label = emotion_name

            print(f"📁 Scanning {emotion_name} folder → Label: {emotion_label}")

            # Find all audio files
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(emotion_folder.glob(f'*{ext}'))

            for audio_file in audio_files:
                dataset.append((str(audio_file), emotion_label))

            print(f"   Found {len(audio_files)} files")

    print(f"\n✅ Total files loaded: {len(dataset)}")
    print(f"📊 Emotion distribution:")

    # Count emotions
    emotion_counts = {}
    for _, emotion in dataset:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count}")

    return dataset


def extract_features_from_dataset(dataset, feature_extractor):
    """
    Extract MFCC features from all audio files in dataset

    Args:
        dataset (list): List of (audio_path, label) tuples
        feature_extractor (MFCCFeatureExtractor): MFCC extractor

    Returns:
        tuple: (features_array, labels_array)
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
            # Extract features
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

    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\n📊 Feature matrix shape: {X.shape}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features per sample: {X.shape[1]}")

    return X, y


def train_and_save_model(X, y, classifier_type='svm', save_path='models/'):
    """
    Train emotion classifier and save it

    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        classifier_type (str): 'svm', 'rf', or 'knn'
        save_path (str): Directory to save model

    Returns:
        EmotionClassifier: Trained classifier
    """
    print(f"\n{'='*70}")
    print(f"Training {classifier_type.upper()} Classifier")
    print(f"{'='*70}\n")

    # Create classifier
    classifier = EmotionClassifier(classifier_type=classifier_type)

    # Train
    results = classifier.train(X, y, test_size=0.2)

    # Save model
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / f'{classifier_type}_emotion_model.pkl'
    classifier.save_model(str(model_file))

    print(f"\n✅ Model saved to: {model_file}")

    return classifier, results


# ============================================================================
# QUICK START: CREATE SAMPLE TRAINING DATA
# ============================================================================

def create_sample_training_data():
    """
    Create a small sample dataset for testing
    This generates synthetic emotional audio samples
    """
    print(f"\n{'='*70}")
    print("Creating Sample Training Data")
    print(f"{'='*70}\n")

    import librosa
    import soundfile as sf

    # Create sample data folder
    sample_folder = Path("sample_training_data")

    emotions = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'satisfied']

    for emotion in emotions:
        emotion_folder = sample_folder / emotion
        emotion_folder.mkdir(parents=True, exist_ok=True)

        # Generate 5 sample audio files per emotion
        for i in range(5):
            # Generate synthetic audio (simple sine waves with variations)
            duration = 3  # seconds
            sr = 22050

            # Different frequency ranges for different emotions (just for demo)
            if emotion == 'angry':
                freq = np.random.randint(300, 400)
            elif emotion == 'happy':
                freq = np.random.randint(400, 500)
            elif emotion == 'sad':
                freq = np.random.randint(150, 250)
            else:
                freq = np.random.randint(250, 350)

            # Generate tone
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.3 * np.sin(2 * np.pi * freq * t)

            # Add some noise
            audio += 0.05 * np.random.randn(len(audio))

            # Save
            filename = emotion_folder / f"{emotion}_sample_{i+1}.wav"
            sf.write(filename, audio, sr)

        print(f"✅ Created {emotion} samples")

    print(f"\n📁 Sample data created in: {sample_folder}")
    print(f"⚠️  NOTE: This is synthetic data for TESTING ONLY!")
    print(f"   For real emotion recognition, use actual emotional speech datasets.")

    return str(sample_folder)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""

    print(f"\n{'='*70}")
    print("CSR EMOTION RECOGNITION - MODEL TRAINING")
    print(f"{'='*70}\n")

    # Ask user what they want to do
    print("Options:")
    print("1. Create sample training data (for testing)")
    print("2. Train from existing dataset folder")
    print("3. Exit")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == '1':
        # Create sample data
        data_folder = create_sample_training_data()
        print("\nNow training on sample data...")

    elif choice == '2':
        # Use existing folder
        data_folder = input("\nEnter path to your dataset folder: ").strip()

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

    # Train models
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

    # Train each classifier
    results_summary = []
    for clf_type in classifiers_to_train:
        classifier, results = train_and_save_model(X, y, classifier_type=clf_type)
        results_summary.append({
            'type': clf_type,
            'accuracy': results['accuracy'],
            'training_samples': len(X)
        })

    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")

    for result in results_summary:
        print(f"{result['type'].upper()}: Accuracy = {result['accuracy']:.2%} "
              f"(trained on {result['training_samples']} samples)")

    print(f"\n✅ Models saved in: models/")
    print(f"\n🔄 Restart your backend server to use the trained models!")
    print(f"   The system will automatically load the trained models.")


def train_from_labeled_recordings(data_folder, classifiers_to_train=None):
    """
    Train emotion recognition models from labeled dataset folder
    This is a wrapper function for use by setup scripts

    Args:
        data_folder (str): Path to folder containing emotion subfolders
        classifiers_to_train (list): List of classifier types to train ['svm', 'rf', 'knn']
                                      If None, trains all three

    Returns:
        dict: Training results summary
    """
    print(f"\n{'='*70}")
    print("TRAINING EMOTION RECOGNITION MODELS")
    print(f"{'='*70}\n")

    if classifiers_to_train is None:
        classifiers_to_train = ['svm', 'rf', 'knn']

    if not os.path.exists(data_folder):
        print(f"❌ Error: Folder '{data_folder}' not found!")
        return None

    # Load dataset
    dataset = load_dataset_from_folder(data_folder)

    if len(dataset) == 0:
        print("❌ No audio files found! Please check your dataset structure.")
        return None

    # Initialize feature extractor
    print("\nInitializing MFCC Feature Extractor...")
    feature_extractor = MFCCFeatureExtractor()

    # Extract features
    X, y = extract_features_from_dataset(dataset, feature_extractor)

    if len(X) == 0:
        print("❌ No features extracted! Check your audio files.")
        return None

    # Train each classifier
    results_summary = []
    for clf_type in classifiers_to_train:
        classifier, results = train_and_save_model(X, y, classifier_type=clf_type)
        results_summary.append({
            'type': clf_type,
            'accuracy': results['accuracy'],
            'training_samples': len(X)
        })

    # Print summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}\n")

    for result in results_summary:
        print(f"{result['type'].upper()}: Accuracy = {result['accuracy']:.2%} "
              f"(trained on {result['training_samples']} samples)")

    print(f"\n✅ Models saved in: models/")
    print(f"\n🔄 Restart your backend server to use the trained models!")
    print(f"   The system will automatically load the trained models.")

    return results_summary


if __name__ == "__main__":
    main()
