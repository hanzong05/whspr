"""
RAVDESS Dataset Downloader and Organizer
Downloads and prepares RAVDESS for emotion model training
"""

import os
import zipfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm

# ============================================================================
# RAVDESS CONFIGURATION
# ============================================================================

# RAVDESS Audio-Only files on Zenodo
# Full dataset: https://zenodo.org/record/1188976
RAVDESS_PARTS = [
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
]

# RAVDESS Emotion Mapping
# Filename format: 03-01-06-01-02-01-12.wav
#   Position 3: Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',     # neutral
    '02': 'neutral',     # calm → neutral
    '03': 'happy',       # happy
    '04': 'sad',         # sad
    '05': 'angry',       # angry
    '06': 'frustrated',  # fearful → frustrated
    '07': 'frustrated',  # disgust → frustrated
    '08': 'satisfied',   # surprised → satisfied (positive emotion)
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def download_ravdess(download_dir='whspr-be/ravdess_download'):
    """
    Download RAVDESS dataset

    Args:
        download_dir (str): Directory to download files
    """
    print("\n" + "="*70)
    print("DOWNLOADING RAVDESS DATASET")
    print("="*70 + "\n")

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)

    print("📦 RAVDESS Audio-Speech Dataset")
    print("   Size: ~2.5 GB")
    print("   Files: ~1,440 emotional speech recordings")
    print("   This will take 5-15 minutes depending on your connection...\n")

    downloaded_files = []

    for i, url in enumerate(RAVDESS_PARTS, 1):
        filename = url.split('/')[-1]
        output_path = download_dir / filename

        if output_path.exists():
            print(f"✓ Already downloaded: {filename}")
            downloaded_files.append(output_path)
            continue

        print(f"\n📥 Downloading part {i}/{len(RAVDESS_PARTS)}: {filename}")
        try:
            download_url(url, str(output_path))
            downloaded_files.append(output_path)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            print("   You can download manually from: https://zenodo.org/record/1188976")
            return None

    return downloaded_files


# ============================================================================
# ORGANIZE FUNCTIONS
# ============================================================================

def extract_and_organize(zip_files, output_dir='whspr-be/training_data'):
    """
    Extract RAVDESS and organize into emotion folders

    Args:
        zip_files (list): List of zip file paths
        output_dir (str): Output directory for organized files
    """
    print("\n" + "="*70)
    print("EXTRACTING AND ORGANIZING RAVDESS")
    print("="*70 + "\n")

    output_dir = Path(output_dir)

    # Create emotion folders
    emotions = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'satisfied']
    for emotion in emotions:
        (output_dir / emotion).mkdir(parents=True, exist_ok=True)

    # Temporary extraction folder
    temp_extract = Path('whspr-be/temp_ravdess_extract')
    temp_extract.mkdir(exist_ok=True)

    total_organized = 0
    emotion_counts = {emotion: 0 for emotion in emotions}

    for zip_file in zip_files:
        print(f"📦 Extracting: {zip_file.name}...")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)
        except Exception as e:
            print(f"❌ Error extracting {zip_file}: {e}")
            continue

    # Process all extracted WAV files
    print("\n🎵 Processing audio files...")

    wav_files = list(temp_extract.rglob('*.wav'))
    print(f"   Found {len(wav_files)} audio files")

    for wav_file in wav_files:
        # RAVDESS filename format: 03-01-06-01-02-01-12.wav
        # We need the 3rd part (emotion code)
        filename = wav_file.name
        parts = filename.split('-')

        if len(parts) >= 3:
            # Only process speech files (modality=03, vocal channel=01)
            modality = parts[0]
            vocal_channel = parts[1]
            emotion_code = parts[2]

            if modality == '03' and vocal_channel == '01':  # Audio-only speech
                if emotion_code in RAVDESS_EMOTION_MAP:
                    emotion = RAVDESS_EMOTION_MAP[emotion_code]

                    # Copy to appropriate emotion folder
                    dest_folder = output_dir / emotion
                    dest_file = dest_folder / filename

                    shutil.copy2(wav_file, dest_file)
                    emotion_counts[emotion] += 1
                    total_organized += 1

    # Clean up temp folder
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_extract)

    # Summary
    print("\n" + "="*70)
    print("✅ RAVDESS ORGANIZATION COMPLETE!")
    print("="*70)
    print(f"\n📊 Organized {total_organized} audio files:")
    for emotion in emotions:
        count = emotion_counts[emotion]
        print(f"   {emotion:12s}: {count:4d} files")

    print(f"\n📁 Files organized in: {output_dir.absolute()}")

    return output_dir


# ============================================================================
# MAIN SETUP
# ============================================================================

def main():
    """Main setup function"""

    print("\n" + "="*70)
    print("RAVDESS DATASET SETUP FOR CSR EMOTION RECOGNITION")
    print("="*70)

    print("\n📋 This script will:")
    print("   1. Download RAVDESS dataset (~2.5 GB)")
    print("   2. Extract and organize files by emotion")
    print("   3. Prepare data for model training")

    print("\n⚠️  You need ~5 GB free disk space")
    print("   Download will take 5-15 minutes\n")

    proceed = input("Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Setup cancelled.")
        return

    # Step 1: Download
    print("\n" + "="*70)
    print("STEP 1: DOWNLOAD RAVDESS")
    print("="*70)

    zip_files = download_ravdess()

    if not zip_files:
        print("\n❌ Download failed. Please try again or download manually from:")
        print("   https://zenodo.org/record/1188976")
        return

    # Step 2: Extract and organize
    print("\n" + "="*70)
    print("STEP 2: ORGANIZE FILES")
    print("="*70)

    training_dir = extract_and_organize(zip_files)

    # Step 3: Ready to train
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print("\n✅ RAVDESS dataset is ready for training!")
    print(f"\n📁 Training data location: {training_dir.absolute()}")

    print("\n🚀 NEXT STEPS:")
    print("   1. Run the training script:")
    print("      python train_model.py")
    print("\n   2. Choose option 2 (Train from existing dataset)")
    print(f"   3. Enter path: {training_dir.absolute()}")
    print("\n   This will train SVM, Random Forest, and KNN models!")

    # Ask if user wants to train now
    train_now = input("\n\nTrain models now? (y/n): ").strip().lower()
    if train_now == 'y':
        print("\n🤖 Starting training process...")
        print("   (Importing training module...)")
        try:
            from train_model import train_from_labeled_recordings
            train_from_labeled_recordings(str(training_dir))
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   Please run manually: python train_model.py")


if __name__ == "__main__":
    main()
