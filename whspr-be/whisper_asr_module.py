"""
CSR Call Recording - Whisper ASR Transcription Module
Converts call recordings to text with timestamps and metadata
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

try:
    import whisper
    import torch
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install openai-whisper torch")
    sys.exit(1)

warnings.filterwarnings('ignore')


class CSRCallTranscriber:
    """
    Whisper ASR Transcription for CSR Call Recordings
    """
    
    # Available Whisper models (larger = more accurate but slower)
    MODELS = {
        'tiny': 'Fastest, least accurate',
        'base': 'Good balance for real-time',
        'small': 'Better accuracy, moderate speed',
        'medium': 'High accuracy, slower',
        'large': 'Best accuracy, slowest'
    }
    
    def __init__(self, model_size='base', device=None, language='en'):
        """
        Initialize Whisper ASR Transcriber
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device (str): 'cuda' for GPU or 'cpu' (auto-detect if None)
            language (str): Language code ('en' for English, 'es' for Spanish, etc.)
        """
        self.model_size = model_size
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"{'='*70}")
        print(f"CSR Call Transcriber - Whisper ASR")
        print(f"{'='*70}")
        print(f"Model: {model_size} ({self.MODELS.get(model_size, 'Unknown')})")
        print(f"Device: {self.device.upper()}")
        print(f"Language: {language}")
        print(f"{'='*70}\n")
        
        # Load Whisper model
        print("Loading Whisper model... This may take a moment.")
        self.model = whisper.load_model(model_size, device=self.device)
        print("✓ Model loaded successfully!\n")
    
    def transcribe_call(self, audio_path, include_timestamps=True, word_timestamps=False):
        """
        Transcribe CSR call recording
        
        Args:
            audio_path (str): Path to audio file (mp3, wav, m4a, etc.)
            include_timestamps (bool): Include segment timestamps
            word_timestamps (bool): Include word-level timestamps (slower)
            
        Returns:
            dict: Transcription results with metadata
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"[PROCESSING] {Path(audio_path).name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Transcription options
        options = {
            'language': self.language,
            'task': 'transcribe',
            'verbose': False,
            'word_timestamps': word_timestamps
        }
        
        # Perform transcription
        print("Transcribing audio...")
        result = self.model.transcribe(audio_path, **options)
        
        # Process results
        transcription_data = self._process_transcription(result, audio_path)
        
        print(f"\n✓ Transcription completed!")
        print(f"Duration: {transcription_data['duration_formatted']}")
        print(f"Segments: {transcription_data['segment_count']}")
        print(f"Word count: {transcription_data['word_count']}\n")
        
        return transcription_data
    
    def _process_transcription(self, result, audio_path):
        """Process raw Whisper output into structured format"""
        
        # Calculate duration
        duration_seconds = result['segments'][-1]['end'] if result['segments'] else 0
        duration_formatted = str(timedelta(seconds=int(duration_seconds)))
        
        # Extract segments with timestamps
        segments = []
        for seg in result['segments']:
            segment_data = {
                'id': seg['id'],
                'start': seg['start'],
                'end': seg['end'],
                'start_time': self._format_timestamp(seg['start']),
                'end_time': self._format_timestamp(seg['end']),
                'text': seg['text'].strip(),
                'confidence': seg.get('avg_logprob', 0)
            }
            
            # Add word-level timestamps if available
            if 'words' in seg:
                segment_data['words'] = [
                    {
                        'word': w['word'],
                        'start': w['start'],
                        'end': w['end'],
                        'confidence': w.get('probability', 0)
                    }
                    for w in seg['words']
                ]
            
            segments.append(segment_data)
        
        # Compile full transcription data
        transcription_data = {
            'audio_file': str(Path(audio_path).name),
            'audio_path': str(audio_path),
            'transcription_timestamp': datetime.now().isoformat(),
            'model': self.model_size,
            'language': result['language'],
            'duration_seconds': duration_seconds,
            'duration_formatted': duration_formatted,
            'full_text': result['text'].strip(),
            'word_count': len(result['text'].split()),
            'segment_count': len(segments),
            'segments': segments
        }
        
        return transcription_data
    
    def _format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))
    
    def _format_timestamp_detailed(self, seconds):
        """Convert seconds to HH:MM:SS.mmm format"""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = td.microseconds // 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def save_transcription(self, transcription_data, output_dir='transcriptions'):
        """
        Save transcription in multiple formats
        
        Args:
            transcription_data (dict): Transcription results
            output_dir (str): Output directory path
            
        Returns:
            dict: Paths to saved files
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Base filename (from audio filename)
        base_name = Path(transcription_data['audio_file']).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = {}
        
        # 1. Save as JSON (complete data)
        json_file = output_path / f"{base_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        saved_files['json'] = str(json_file)
        print(f"✓ Saved JSON: {json_file}")
        
        # 2. Save as plain text
        txt_file = output_path / f"{base_name}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"CSR Call Transcription\n")
            f.write(f"{'='*70}\n")
            f.write(f"File: {transcription_data['audio_file']}\n")
            f.write(f"Date: {transcription_data['transcription_timestamp']}\n")
            f.write(f"Duration: {transcription_data['duration_formatted']}\n")
            f.write(f"Language: {transcription_data['language']}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"FULL TRANSCRIPTION:\n\n")
            f.write(transcription_data['full_text'])
        saved_files['txt'] = str(txt_file)
        print(f"✓ Saved TXT: {txt_file}")
        
        # 3. Save as timestamped transcript
        timestamped_file = output_path / f"{base_name}_{timestamp}_timestamped.txt"
        with open(timestamped_file, 'w', encoding='utf-8') as f:
            f.write(f"CSR Call Transcription (Timestamped)\n")
            f.write(f"{'='*70}\n\n")
            for seg in transcription_data['segments']:
                f.write(f"[{seg['start_time']} - {seg['end_time']}]\n")
                f.write(f"{seg['text']}\n\n")
        saved_files['timestamped_txt'] = str(timestamped_file)
        print(f"✓ Saved Timestamped TXT: {timestamped_file}")
        
        # 4. Save as SRT (subtitle format)
        srt_file = output_path / f"{base_name}_{timestamp}.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(transcription_data['segments'], 1):
                start_srt = self._format_timestamp_detailed(seg['start']).replace('.', ',')
                end_srt = self._format_timestamp_detailed(seg['end']).replace('.', ',')
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{seg['text']}\n\n")
        saved_files['srt'] = str(srt_file)
        print(f"✓ Saved SRT: {srt_file}")
        
        # 5. Save metadata summary
        summary_file = output_path / f"{base_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"CSR CALL SUMMARY\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Audio File: {transcription_data['audio_file']}\n")
            f.write(f"Transcription Date: {transcription_data['transcription_timestamp']}\n")
            f.write(f"Model Used: Whisper {transcription_data['model']}\n")
            f.write(f"Language Detected: {transcription_data['language']}\n")
            f.write(f"Call Duration: {transcription_data['duration_formatted']}\n")
            f.write(f"Total Segments: {transcription_data['segment_count']}\n")
            f.write(f"Word Count: {transcription_data['word_count']}\n")
            f.write(f"\nFirst 200 characters:\n")
            f.write(f"{transcription_data['full_text'][:200]}...\n")
        saved_files['summary'] = str(summary_file)
        print(f"✓ Saved Summary: {summary_file}")
        
        return saved_files
    
    def display_transcription(self, transcription_data):
        """Display transcription in console"""
        print(f"\n{'='*70}")
        print(f"TRANSCRIPTION RESULTS")
        print(f"{'='*70}")
        print(f"File: {transcription_data['audio_file']}")
        print(f"Duration: {transcription_data['duration_formatted']}")
        print(f"Language: {transcription_data['language']}")
        print(f"Segments: {transcription_data['segment_count']}")
        print(f"Words: {transcription_data['word_count']}")
        print(f"{'='*70}\n")
        
        print(f"FULL TRANSCRIPTION:")
        print(f"{'-'*70}")
        print(transcription_data['full_text'])
        print(f"{'-'*70}\n")
        
        print(f"SEGMENTS WITH TIMESTAMPS:")
        print(f"{'-'*70}")
        for seg in transcription_data['segments']:
            print(f"[{seg['start_time']} → {seg['end_time']}]")
            print(f"  {seg['text']}\n")
        print(f"{'='*70}\n")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CSR Call Recording - Whisper ASR Transcription',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python whisper_asr_module.py call_recording.wav
  
  # Use better model for accuracy
  python whisper_asr_module.py call_recording.wav --model medium
  
  # Save to specific directory
  python whisper_asr_module.py call_recording.wav --output transcripts/
  
  # Include word-level timestamps
  python whisper_asr_module.py call_recording.wav --word-timestamps
        """
    )
    
    parser.add_argument(
        'audio_file',
        help='Path to CSR call recording (mp3, wav, m4a, etc.)'
    )
    
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--language',
        default='en',
        help='Language code (default: en for English)'
    )
    
    parser.add_argument(
        '--output',
        default='transcriptions',
        help='Output directory for transcription files'
    )
    
    parser.add_argument(
        '--word-timestamps',
        action='store_true',
        help='Include word-level timestamps (slower)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Display only, do not save files'
    )
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = CSRCallTranscriber(
        model_size=args.model,
        language=args.language
    )
    
    # Transcribe call
    result = transcriber.transcribe_call(
        args.audio_file,
        word_timestamps=args.word_timestamps
    )
    
    # Display results
    transcriber.display_transcription(result)
    
    # Save results
    if not args.no_save:
        saved_files = transcriber.save_transcription(result, args.output)
        print(f"\n{'='*70}")
        print(f"Files saved to: {args.output}/")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()