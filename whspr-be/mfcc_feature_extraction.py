"""
//mfcc_feature_extraction.py
CSR Call Recording - MFCC Feature Extraction Module
Extracts audio features for emotion analysis and classification
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import librosa
    import librosa.display
except ImportError:
    print("Error: librosa not installed.")
    print("Install with: pip install librosa")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be disabled.")

warnings.filterwarnings('ignore')


class MFCCFeatureExtractor:
    """
    Extract MFCC and other audio features for emotion classification
    """

    def __init__(self,
                 n_mfcc=40,
                 n_fft=2048,
                 hop_length=512,
                 n_mels=128,
                 sample_rate=22050):
        """
        Initialize MFCC Feature Extractor

        Args:
            n_mfcc (int): Number of MFCC coefficients to extract
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            n_mels (int): Number of Mel bands
            sample_rate (int): Target sample rate for audio
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        print(f"{'='*70}")
        print(f"MFCC Feature Extractor Initialized")
        print(f"{'='*70}")
        print(f"MFCC Coefficients: {n_mfcc}")
        print(f"FFT Window Size: {n_fft}")
        print(f"Hop Length: {hop_length}")
        print(f"Mel Bands: {n_mels}")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"{'='*70}\n")

    def load_audio(self, audio_path):
        """
        Load and preprocess audio file

        Args:
            audio_path (str): Path to audio file

        Returns:
            tuple: (audio_data, sample_rate)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[Loading Audio] {Path(audio_path).name}")

        # Load audio with librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        duration = librosa.get_duration(y=audio, sr=sr)

        print(f"✓ Audio loaded successfully")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  Samples: {len(audio)}\n")

        return audio, sr

    def extract_mfcc(self, audio, sr):
        """
        Extract MFCC features from audio

        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate

        Returns:
            dict: MFCC features and statistics
        """
        print("[Extracting MFCC] Computing Mel-Frequency Cepstral Coefficients...")

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Calculate statistics for each MFCC coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)

        # Delta MFCCs (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)

        # Delta-Delta MFCCs (second derivative)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

        print(f"✓ MFCC extraction complete")
        print(
            f"  Shape: {mfccs.shape} ({self.n_mfcc} coefficients × {mfccs.shape[1]} frames)")
        print(
            f"  Mean MFCC range: [{float(mfcc_mean.min()):.2f}, {float(mfcc_mean.max()):.2f}]\n")

        return {
            'mfcc': mfccs,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_max': mfcc_max,
            'mfcc_min': mfcc_min,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta_mean': mfcc_delta_mean,
            'mfcc_delta_std': mfcc_delta_std,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_delta2_mean': mfcc_delta2_mean,
            'mfcc_delta2_std': mfcc_delta2_std
        }

    def extract_spectral_features(self, audio, sr):
        """
        Extract spectral features for emotion analysis

        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate

        Returns:
            dict: Spectral features
        """
        print("[Extracting Spectral Features] Computing additional audio features...")

        # Spectral Centroid - brightness of sound
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Spectral Rolloff - frequency below which specified percentage of total spectral energy
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Spectral Bandwidth - width of frequency band
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        # Zero Crossing Rate - how often signal changes sign
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]

        # Chroma features - pitch class representation
        chroma_stft = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )

        # Root Mean Square Energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]

        print(f"✓ Spectral features extracted")
        print(f"  Spectral Centroid: {float(np.mean(spectral_centroids)):.2f} Hz")
        print(f"  Zero Crossing Rate: {float(np.mean(zero_crossing_rate)):.4f}")
        print(f"  RMS Energy: {float(np.mean(rms)):.4f}\n")

        return {
            'spectral_centroid': spectral_centroids,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff': spectral_rolloff,
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zero_crossing_rate': zero_crossing_rate,
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate),
            'chroma_stft': chroma_stft,
            'chroma_stft_mean': np.mean(chroma_stft, axis=1),
            'chroma_stft_std': np.std(chroma_stft, axis=1),
            'mel_spectrogram': mel_spectrogram,
            'rms': rms,
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms)
        }

    def extract_prosodic_features(self, audio, sr):
        """
        Extract prosodic features (pitch, tempo, energy)

        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate

        Returns:
            dict: Prosodic features
        """
        print("[Extracting Prosodic Features] Computing pitch, tempo, and energy...")

        # Pitch (F0) estimation using YIN algorithm
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Get the most prominent pitch at each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Filter out zero values
                pitch_values.append(pitch)

        pitch_values = np.array(
            pitch_values) if pitch_values else np.array([0])

        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

        # Energy contour
        energy = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]

        print(f"✓ Prosodic features extracted")
        print(f"  Mean Pitch: {float(np.mean(pitch_values)):.2f} Hz")
        print(f"  Pitch Std: {float(np.std(pitch_values)):.2f} Hz")
        print(f"  Tempo: {float(tempo):.2f} BPM\n")

        return {
            'pitch_values': pitch_values,
            'pitch_mean': np.mean(pitch_values),
            'pitch_std': np.std(pitch_values),
            'pitch_max': np.max(pitch_values),
            'pitch_min': np.min(pitch_values),
            'tempo': float(tempo),
            'energy': energy,
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy)
        }

    def create_feature_vector(self, mfcc_features, spectral_features, prosodic_features):
        """
        Create a single feature vector for ML classification

        Args:
            mfcc_features (dict): MFCC features
            spectral_features (dict): Spectral features
            prosodic_features (dict): Prosodic features

        Returns:
            np.array: Feature vector for ML model
        """
        print("[Creating Feature Vector] Combining all features...")

        feature_vector = np.concatenate([
            # MFCC statistics (40 × 6 = 240 features)
            mfcc_features['mfcc_mean'],
            mfcc_features['mfcc_std'],
            mfcc_features['mfcc_max'],
            mfcc_features['mfcc_min'],
            mfcc_features['mfcc_delta_mean'],
            mfcc_features['mfcc_delta_std'],
            mfcc_features['mfcc_delta2_mean'],
            mfcc_features['mfcc_delta2_std'],

            # Spectral features (13 features)
            [spectral_features['spectral_centroid_mean']],
            [spectral_features['spectral_centroid_std']],
            [spectral_features['spectral_rolloff_mean']],
            [spectral_features['spectral_rolloff_std']],
            [spectral_features['spectral_bandwidth_mean']],
            [spectral_features['spectral_bandwidth_std']],
            [spectral_features['zero_crossing_rate_mean']],
            [spectral_features['zero_crossing_rate_std']],
            spectral_features['chroma_stft_mean'],
            [spectral_features['rms_mean']],
            [spectral_features['rms_std']],

            # Prosodic features (6 features)
            [prosodic_features['pitch_mean']],
            [prosodic_features['pitch_std']],
            [prosodic_features['pitch_max']],
            [prosodic_features['pitch_min']],
            [prosodic_features['tempo']],
            [prosodic_features['energy_mean']],
            [prosodic_features['energy_std']]
        ])

        print(f"✓ Feature vector created")
        print(f"  Total features: {len(feature_vector)}")
        print(
            f"  Feature range: [{float(feature_vector.min()):.2f}, {float(feature_vector.max()):.2f}]\n")

        return feature_vector

    def extract_features(self, audio_path):
        """
        Extract feature vector from audio file (for ML model predictions)

        Args:
            audio_path (str): Path to audio file

        Returns:
            np.array: Feature vector for ML model, or None if extraction fails
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Extract MFCC features
            mfcc_features = self.extract_mfcc(audio, sr)

            # Extract spectral features
            spectral_features = self.extract_spectral_features(audio, sr)

            # Extract prosodic features
            prosodic_features = self.extract_prosodic_features(audio, sr)

            # Create feature vector for ML
            feature_vector = self.create_feature_vector(
                mfcc_features, spectral_features, prosodic_features
            )

            return feature_vector

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

    def extract_all_features(self, audio_path):
        """
        Extract all features from audio file

        Args:
            audio_path (str): Path to audio file

        Returns:
            dict: All extracted features
        """
        print(f"\n{'='*70}")
        print(f"FEATURE EXTRACTION PIPELINE")
        print(f"{'='*70}\n")

        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio, sr)

        # Extract spectral features
        spectral_features = self.extract_spectral_features(audio, sr)

        # Extract prosodic features
        prosodic_features = self.extract_prosodic_features(audio, sr)

        # Create feature vector for ML
        feature_vector = self.create_feature_vector(
            mfcc_features, spectral_features, prosodic_features
        )

        # Compile all results
        results = {
            'audio_file': str(Path(audio_path).name),
            'audio_path': str(audio_path),
            'extraction_timestamp': datetime.now().isoformat(),
            'duration_seconds': float(librosa.get_duration(y=audio, sr=sr)),
            'sample_rate': int(sr),
            'feature_vector': feature_vector.tolist(),
            'feature_vector_length': len(feature_vector),
            'mfcc_features': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in mfcc_features.items()
            },
            'spectral_features': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in spectral_features.items()
            },
            'prosodic_features': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in prosodic_features.items()
            }
        }

        print(f"{'='*70}")
        print(f"✓ FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}\n")

        return results

    def visualize_features(self, audio_path, output_dir='visualizations'):
        """
        Create visualizations of extracted features

        Args:
            audio_path (str): Path to audio file
            output_dir (str): Directory to save visualizations

        Returns:
            dict: Paths to saved visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return {}

        print(f"[Visualizing Features] Creating plots...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio, sr = self.load_audio(audio_path)

        base_name = Path(audio_path).stem
        saved_plots = {}

        # 1. Waveform
        plt.figure(figsize=(14, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        waveform_path = output_path / f"{base_name}_waveform.png"
        plt.savefig(waveform_path, dpi=150)
        plt.close()
        saved_plots['waveform'] = str(waveform_path)

        # 2. MFCC Heatmap
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        plt.figure(figsize=(14, 6))
        librosa.display.specshow(mfccs, x_axis='time',
                                 sr=sr, hop_length=self.hop_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features')
        plt.tight_layout()
        mfcc_path = output_path / f"{base_name}_mfcc.png"
        plt.savefig(mfcc_path, dpi=150)
        plt.close()
        saved_plots['mfcc'] = str(mfcc_path)

        # 3. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(14, 6))
        librosa.display.specshow(
            mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=self.hop_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        mel_path = output_path / f"{base_name}_mel_spectrogram.png"
        plt.savefig(mel_path, dpi=150)
        plt.close()
        saved_plots['mel_spectrogram'] = str(mel_path)

        # 4. Spectral Features
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[
            0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=self.hop_length)

        axes[0].plot(t, spectral_centroids, color='b', alpha=0.7)
        axes[0].set_title('Spectral Centroid')
        axes[0].set_ylabel('Hz')

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        axes[1].plot(t, spectral_rolloff, color='g', alpha=0.7)
        axes[1].set_title('Spectral Rolloff')
        axes[1].set_ylabel('Hz')

        zero_crossing = librosa.feature.zero_crossing_rate(y=audio)[0]
        axes[2].plot(t, zero_crossing, color='r', alpha=0.7)
        axes[2].set_title('Zero Crossing Rate')
        axes[2].set_xlabel('Time (seconds)')

        plt.tight_layout()
        spectral_path = output_path / f"{base_name}_spectral_features.png"
        plt.savefig(spectral_path, dpi=150)
        plt.close()
        saved_plots['spectral_features'] = str(spectral_path)

        print(f"✓ Visualizations saved to: {output_dir}/\n")

        return saved_plots

    def save_features(self, features, output_dir='features'):
        """
        Save extracted features to files

        Args:
            features (dict): Extracted features
            output_dir (str): Output directory

        Returns:
            dict: Paths to saved files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = Path(features['audio_file']).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        saved_files = {}

        # 1. Save as JSON
        json_file = output_path / f"{base_name}_{timestamp}_features.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2)
        saved_files['json'] = str(json_file)
        print(f"✓ Saved JSON: {json_file}")

        # 2. Save feature vector as CSV (for ML models)
        csv_file = output_path / f"{base_name}_{timestamp}_feature_vector.csv"
        df = pd.DataFrame([features['feature_vector']])
        df.to_csv(csv_file, index=False, header=False)
        saved_files['csv'] = str(csv_file)
        print(f"✓ Saved CSV: {csv_file}")

        # 3. Save feature vector as NumPy array
        npy_file = output_path / f"{base_name}_{timestamp}_feature_vector.npy"
        np.save(npy_file, np.array(features['feature_vector']))
        saved_files['npy'] = str(npy_file)
        print(f"✓ Saved NPY: {npy_file}")

        # 4. Save summary
        summary_file = output_path / f"{base_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"MFCC FEATURE EXTRACTION SUMMARY\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Audio File: {features['audio_file']}\n")
            f.write(f"Extraction Time: {features['extraction_timestamp']}\n")
            f.write(f"Duration: {features['duration_seconds']:.2f} seconds\n")
            f.write(f"Sample Rate: {features['sample_rate']} Hz\n")
            f.write(
                f"Feature Vector Length: {features['feature_vector_length']}\n\n")

            f.write(f"MFCC Features:\n")
            f.write(
                f"  Mean MFCC range: [{min(features['mfcc_features']['mfcc_mean']):.2f}, {max(features['mfcc_features']['mfcc_mean']):.2f}]\n\n")

            f.write(f"Spectral Features:\n")
            f.write(
                f"  Spectral Centroid: {features['spectral_features']['spectral_centroid_mean']:.2f} Hz\n")
            f.write(
                f"  Zero Crossing Rate: {features['spectral_features']['zero_crossing_rate_mean']:.4f}\n")
            f.write(
                f"  RMS Energy: {features['spectral_features']['rms_mean']:.4f}\n\n")

            f.write(f"Prosodic Features:\n")
            f.write(
                f"  Mean Pitch: {features['prosodic_features']['pitch_mean']:.2f} Hz\n")
            f.write(
                f"  Tempo: {features['prosodic_features']['tempo']:.2f} BPM\n")

        saved_files['summary'] = str(summary_file)
        print(f"✓ Saved Summary: {summary_file}\n")

        return saved_files


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='CSR Call Recording - MFCC Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic feature extraction
  python mfcc_feature_extraction.py call_recording.wav
  
  # Extract and visualize
  python mfcc_feature_extraction.py call_recording.wav --visualize
  
  # Custom output directory
  python mfcc_feature_extraction.py call_recording.wav --output features/
  
  # Custom MFCC parameters
  python mfcc_feature_extraction.py call_recording.wav --n-mfcc 20
        """
    )

    parser.add_argument(
        'audio_file',
        help='Path to audio file (wav, mp3, etc.)'
    )

    parser.add_argument(
        '--n-mfcc',
        type=int,
        default=40,
        help='Number of MFCC coefficients (default: 40)'
    )

    parser.add_argument(
        '--output',
        default='features',
        help='Output directory for feature files'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create feature visualizations'
    )

    parser.add_argument(
        '--viz-output',
        default='visualizations',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = MFCCFeatureExtractor(n_mfcc=args.n_mfcc)

    # Extract features
    features = extractor.extract_all_features(args.audio_file)

    # Save features
    saved_files = extractor.save_features(features, args.output)

    # Visualize if requested
    if args.visualize:
        viz_files = extractor.visualize_features(
            args.audio_file, args.viz_output)
        print(f"Visualizations saved to: {args.viz_output}/")

    print(f"\n{'='*70}")
    print(f"✓ Feature extraction complete!")
    print(f"{'='*70}")
    print(f"Output directory: {args.output}/")
    print(f"Feature vector ready for ML classification")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
