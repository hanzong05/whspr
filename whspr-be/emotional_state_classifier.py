"""
//emotional_state_classifier.py
CSR Call Recording - Emotional State Classification Module
Advanced emotion analysis with multi-dimensional classification
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EmotionalStateClassifier:
    """
    Advanced Emotional State Classification System
    Analyzes emotions with context, intensity, and temporal patterns
    """

    # Primary emotions with descriptions
    EMOTIONS = {
        'angry': {
            'name': 'Angry',
            'description': 'Customer is frustrated, upset, or irritated',
            'valence': 'negative',
            'arousal': 'high',
            'severity': 'high',
            'indicators': ['raised voice', 'fast speech', 'high pitch variation']
        },
        'frustrated': {
            'name': 'Frustrated',
            'description': 'Customer is annoyed but not fully angry',
            'valence': 'negative',
            'arousal': 'medium',
            'severity': 'medium',
            'indicators': ['repetitive statements', 'sighing', 'medium pace']
        },
        'sad': {
            'name': 'Sad',
            'description': 'Customer is disappointed or unhappy',
            'valence': 'negative',
            'arousal': 'low',
            'severity': 'medium',
            'indicators': ['low energy', 'slow speech', 'monotone']
        },
        'neutral': {
            'name': 'Neutral',
            'description': 'Customer is calm and emotionally balanced',
            'valence': 'neutral',
            'arousal': 'low',
            'severity': 'low',
            'indicators': ['steady pace', 'normal pitch', 'balanced tone']
        },
        'satisfied': {
            'name': 'Satisfied',
            'description': 'Customer is content and pleased',
            'valence': 'positive',
            'arousal': 'low',
            'severity': 'low',
            'indicators': ['calm tone', 'positive words', 'acknowledgment']
        },
        'happy': {
            'name': 'Happy',
            'description': 'Customer is pleased and enthusiastic',
            'valence': 'positive',
            'arousal': 'high',
            'severity': 'low',
            'indicators': ['upbeat tone', 'faster speech', 'positive expressions']
        }
    }

    # Emotion intensity levels
    INTENSITY_LEVELS = {
        'very_low': (0.0, 0.3),
        'low': (0.3, 0.5),
        'medium': (0.5, 0.7),
        'high': (0.7, 0.85),
        'very_high': (0.85, 1.0)
    }

    # Risk levels based on emotion
    RISK_LEVELS = {
        'critical': ['angry'],
        'high': ['frustrated'],
        'medium': ['sad'],
        'low': ['neutral', 'satisfied', 'happy']
    }

    def __init__(self):
        """Initialize Emotional State Classifier"""
        print(f"{'='*70}")
        print(f"Emotional State Classifier Initialized")
        print(f"{'='*70}")
        print(f"Supported Emotions: {len(self.EMOTIONS)}")
        print(f"  {', '.join([e['name'] for e in self.EMOTIONS.values()])}")
        print(f"{'='*70}\n")

    def classify_emotional_state(self, prediction_result, audio_features=None, transcription=None):
        """
        Classify emotional state with detailed analysis

        Args:
            prediction_result (dict): ML classifier prediction output
            audio_features (dict): MFCC and audio features
            transcription (dict): Whisper ASR transcription

        Returns:
            dict: Detailed emotional state classification
        """
        emotion = prediction_result['emotion']
        confidence = prediction_result['confidence']
        all_probabilities = prediction_result['all_probabilities']

        # Get emotion details
        emotion_info = self.EMOTIONS.get(emotion, {
            'name': emotion.capitalize(),
            'valence': 'unknown',
            'arousal': 'unknown',
            'severity': 'unknown'
        })

        # Determine intensity
        intensity = self._calculate_intensity(confidence)

        # Determine risk level
        risk_level = self._calculate_risk_level(emotion, confidence)

        # Analyze valence (positive/negative/neutral)
        valence_score = self._calculate_valence_score(all_probabilities)

        # Analyze arousal (energy level)
        arousal_score = self._calculate_arousal_score(
            all_probabilities, audio_features)

        # Get secondary emotions
        secondary_emotions = self._get_secondary_emotions(
            all_probabilities, threshold=0.15)

        # Analyze emotion stability
        stability = self._analyze_emotion_stability(all_probabilities)

        # Create emotional state profile
        emotional_state = {
            'timestamp': datetime.now().isoformat(),
            'primary_emotion': {
                'emotion': emotion,
                'name': emotion_info['name'],
                'confidence': float(confidence),
                'intensity': intensity,
                'description': emotion_info.get('description', '')
            },
            'secondary_emotions': secondary_emotions,
            'emotional_dimensions': {
                'valence': {
                    'value': valence_score,
                    'category': self._categorize_valence(valence_score),
                    'description': 'Positive-Negative spectrum'
                },
                'arousal': {
                    'value': arousal_score,
                    'category': self._categorize_arousal(arousal_score),
                    'description': 'Energy/Activation level'
                }
            },
            'risk_assessment': {
                'level': risk_level,
                'priority': self._get_priority_level(risk_level),
                'requires_escalation': risk_level in ['critical', 'high']
            },
            'stability': {
                'score': stability,
                'is_stable': stability > 0.6,
                'description': 'Emotional consistency'
            },
            'all_emotion_probabilities': all_probabilities
        }

        # Add audio-based insights if available
        if audio_features:
            emotional_state['audio_indicators'] = self._analyze_audio_indicators(
                emotion, audio_features
            )

        # Add text-based insights if available
        if transcription:
            emotional_state['transcription_indicators'] = self._analyze_transcription_indicators(
                emotion, transcription
            )

        return emotional_state

    def _calculate_intensity(self, confidence):
        """Calculate emotion intensity based on confidence"""
        for level, (min_val, max_val) in self.INTENSITY_LEVELS.items():
            if min_val <= confidence < max_val:
                return {
                    'level': level,
                    'score': float(confidence),
                    'description': f"{level.replace('_', ' ').title()}"
                }
        return {
            'level': 'very_high',
            'score': float(confidence),
            'description': 'Very High'
        }

    def _calculate_risk_level(self, emotion, confidence):
        """Calculate risk level for intervention"""
        for risk, emotions in self.RISK_LEVELS.items():
            if emotion in emotions:
                # Adjust risk based on confidence
                if confidence > 0.8:
                    return risk
                elif confidence > 0.6:
                    # Lower risk by one level if confidence is moderate
                    risk_order = ['critical', 'high', 'medium', 'low']
                    idx = risk_order.index(risk)
                    return risk_order[min(idx + 1, len(risk_order) - 1)]
                else:
                    return 'medium'
        return 'low'

    def _calculate_valence_score(self, probabilities):
        """
        Calculate valence score (-1 to 1)
        -1: very negative, 0: neutral, 1: very positive
        """
        positive_emotions = ['happy', 'satisfied']
        negative_emotions = ['angry', 'frustrated', 'sad']

        positive_score = sum(probabilities.get(e, 0)
                             for e in positive_emotions)
        negative_score = sum(probabilities.get(e, 0)
                             for e in negative_emotions)

        valence = positive_score - negative_score
        return float(np.clip(valence, -1, 1))

    def _calculate_arousal_score(self, probabilities, audio_features=None):
        """
        Calculate arousal score (0 to 1)
        0: very low energy, 1: very high energy
        """
        high_arousal = ['angry', 'happy']
        low_arousal = ['sad', 'neutral', 'satisfied']

        high_score = sum(probabilities.get(e, 0) for e in high_arousal)

        # Enhance with audio features if available
        if audio_features and 'prosodic_features' in audio_features:
            tempo = audio_features['prosodic_features'].get('tempo', 120)
            energy = audio_features['prosodic_features'].get(
                'energy_mean', 0.5)

            # Normalize tempo (typical speech: 100-180 BPM)
            tempo_score = np.clip((tempo - 100) / 80, 0, 1)

            # Combine emotion-based and audio-based arousal
            arousal = (high_score * 0.6) + (tempo_score * 0.2) + (energy * 0.2)
        else:
            arousal = high_score

        return float(np.clip(arousal, 0, 1))

    def _categorize_valence(self, valence_score):
        """Categorize valence score"""
        if valence_score > 0.3:
            return 'Positive'
        elif valence_score < -0.3:
            return 'Negative'
        else:
            return 'Neutral'

    def _categorize_arousal(self, arousal_score):
        """Categorize arousal score"""
        if arousal_score > 0.7:
            return 'High Energy'
        elif arousal_score > 0.3:
            return 'Medium Energy'
        else:
            return 'Low Energy'

    def _get_secondary_emotions(self, probabilities, threshold=0.15):
        """Get secondary emotions above threshold"""
        # Sort by probability
        sorted_emotions = sorted(
            probabilities.items(), key=lambda x: x[1], reverse=True)

        # Skip primary (first) and get secondary above threshold
        secondary = []
        for emotion, prob in sorted_emotions[1:]:
            if prob >= threshold:
                secondary.append({
                    'emotion': emotion,
                    'probability': float(prob),
                    'name': self.EMOTIONS.get(emotion, {}).get('name', emotion.capitalize())
                })

        return secondary

    def _analyze_emotion_stability(self, probabilities):
        """
        Analyze how stable/clear the emotion is
        High stability: one dominant emotion
        Low stability: mixed emotions
        """
        probs = list(probabilities.values())
        max_prob = max(probs)

        # Calculate entropy (measure of uncertainty)
        probs_array = np.array(probs)
        probs_array = probs_array[probs_array > 0]  # Remove zeros
        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))

        # Normalize entropy (max entropy for 6 emotions ≈ 2.58)
        max_entropy = np.log2(len(self.EMOTIONS))
        normalized_entropy = entropy / max_entropy

        # Stability is inverse of normalized entropy
        stability = 1 - normalized_entropy

        return float(stability)

    def _get_priority_level(self, risk_level):
        """Get priority level for response"""
        priority_map = {
            'critical': 'P1 - Immediate',
            'high': 'P2 - Urgent',
            'medium': 'P3 - Normal',
            'low': 'P4 - Low'
        }
        return priority_map.get(risk_level, 'P3 - Normal')

    def _analyze_audio_indicators(self, emotion, audio_features):
        """Analyze audio features for emotion indicators"""
        indicators = []

        if 'prosodic_features' not in audio_features:
            return indicators

        prosodic = audio_features['prosodic_features']
        spectral = audio_features.get('spectral_features', {})

        # Pitch analysis
        pitch_mean = prosodic.get('pitch_mean', 0)
        pitch_std = prosodic.get('pitch_std', 0)

        if pitch_mean > 200:
            indicators.append({
                'feature': 'High Pitch',
                'value': f"{pitch_mean:.1f} Hz",
                'interpretation': 'Elevated vocal tension or excitement'
            })
        elif pitch_mean < 100:
            indicators.append({
                'feature': 'Low Pitch',
                'value': f"{pitch_mean:.1f} Hz",
                'interpretation': 'Low energy or sadness'
            })

        if pitch_std > 50:
            indicators.append({
                'feature': 'High Pitch Variation',
                'value': f"{pitch_std:.1f} Hz",
                'interpretation': 'Emotional expressiveness or agitation'
            })

        # Tempo analysis
        tempo = prosodic.get('tempo', 0)
        if tempo > 150:
            indicators.append({
                'feature': 'Fast Tempo',
                'value': f"{tempo:.1f} BPM",
                'interpretation': 'High energy, excitement, or anxiety'
            })
        elif tempo < 100:
            indicators.append({
                'feature': 'Slow Tempo',
                'value': f"{tempo:.1f} BPM",
                'interpretation': 'Low energy or sadness'
            })

        # Energy analysis
        energy = prosodic.get('energy_mean', 0)
        if energy > 0.7:
            indicators.append({
                'feature': 'High Energy',
                'value': f"{energy:.2f}",
                'interpretation': 'Strong emotional expression'
            })
        elif energy < 0.3:
            indicators.append({
                'feature': 'Low Energy',
                'value': f"{energy:.2f}",
                'interpretation': 'Fatigue or resignation'
            })

        # Spectral centroid (brightness)
        if spectral:
            centroid = spectral.get('spectral_centroid_mean', 0)
            if centroid > 3000:
                indicators.append({
                    'feature': 'Bright Voice',
                    'value': f"{centroid:.0f} Hz",
                    'interpretation': 'Tension or high arousal'
                })

        return indicators

    def _analyze_transcription_indicators(self, emotion, transcription):
        """Analyze transcription for emotion indicators"""
        indicators = []

        text = transcription.get('text', '').lower()

        # Negative keywords
        negative_words = ['angry', 'frustrated', 'upset', 'terrible', 'awful', 'horrible',
                          'unacceptable', 'disappointed', 'annoyed', 'ridiculous']
        negative_count = sum(1 for word in negative_words if word in text)

        if negative_count > 0:
            indicators.append({
                'feature': 'Negative Language',
                'value': f"{negative_count} negative words detected",
                'interpretation': 'Customer expressing dissatisfaction'
            })

        # Positive keywords
        positive_words = ['thank', 'thanks', 'great', 'excellent', 'wonderful', 'perfect',
                          'appreciate', 'satisfied', 'happy', 'pleased']
        positive_count = sum(1 for word in positive_words if word in text)

        if positive_count > 0:
            indicators.append({
                'feature': 'Positive Language',
                'value': f"{positive_count} positive words detected",
                'interpretation': 'Customer expressing satisfaction'
            })

        # Urgency indicators
        urgency_words = ['urgent', 'immediately',
                         'asap', 'now', 'quickly', 'hurry']
        urgency_count = sum(1 for word in urgency_words if word in text)

        if urgency_count > 0:
            indicators.append({
                'feature': 'Urgency Markers',
                'value': f"{urgency_count} urgency indicators",
                'interpretation': 'High priority or time-sensitive issue'
            })

        # Question marks (seeking help)
        question_count = text.count('?') if '?' in text else 0
        if question_count > 3:
            indicators.append({
                'feature': 'Multiple Questions',
                'value': f"{question_count} questions",
                'interpretation': 'Seeking clarification or confused'
            })

        return indicators

    def classify_call_segments(self, segment_predictions):
        """
        Classify emotions across call segments for temporal analysis

        Args:
            segment_predictions (list): List of predictions for each segment

        Returns:
            dict: Temporal emotion analysis
        """
        if not segment_predictions:
            return {}

        # Track emotion changes
        emotion_timeline = []
        emotion_transitions = []

        prev_emotion = None
        for i, pred in enumerate(segment_predictions):
            emotion = pred['emotion']
            confidence = pred['confidence']

            emotion_timeline.append({
                'segment': i,
                'emotion': emotion,
                'confidence': confidence
            })

            # Track transitions
            if prev_emotion and prev_emotion != emotion:
                emotion_transitions.append({
                    'from': prev_emotion,
                    'to': emotion,
                    'segment': i
                })

            prev_emotion = emotion

        # Calculate dominant emotion
        emotions = [p['emotion'] for p in segment_predictions]
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0]

        # Calculate emotion volatility
        volatility = len(emotion_transitions) / \
            len(segment_predictions) if segment_predictions else 0

        # Determine overall trajectory
        first_emotion = emotions[0] if emotions else None
        last_emotion = emotions[-1] if emotions else None

        trajectory = self._determine_trajectory(first_emotion, last_emotion)

        temporal_analysis = {
            'emotion_timeline': emotion_timeline,
            'dominant_emotion': {
                'emotion': dominant_emotion[0],
                'count': dominant_emotion[1],
                'percentage': (dominant_emotion[1] / len(emotions)) * 100
            },
            'emotion_distribution': dict(emotion_counts),
            'transitions': emotion_transitions,
            'volatility': {
                'score': float(volatility),
                'level': 'High' if volatility > 0.5 else 'Medium' if volatility > 0.2 else 'Low'
            },
            'trajectory': trajectory,
            'total_segments': len(segment_predictions)
        }

        return temporal_analysis

    def _determine_trajectory(self, first_emotion, last_emotion):
        """Determine emotional trajectory of the call"""
        if not first_emotion or not last_emotion:
            return 'Unknown'

        first_valence = self.EMOTIONS.get(
            first_emotion, {}).get('valence', 'neutral')
        last_valence = self.EMOTIONS.get(
            last_emotion, {}).get('valence', 'neutral')

        if first_valence == 'negative' and last_valence == 'positive':
            return 'Improving (Negative → Positive)'
        elif first_valence == 'positive' and last_valence == 'negative':
            return 'Declining (Positive → Negative)'
        elif first_valence == 'negative' and last_valence == 'negative':
            return 'Persistently Negative'
        elif first_valence == 'positive' and last_valence == 'positive':
            return 'Persistently Positive'
        else:
            return 'Stable/Mixed'

    def generate_emotion_report(self, emotional_state):
        """Generate human-readable emotion report"""
        report = []

        report.append(f"{'='*70}")
        report.append(f"EMOTIONAL STATE ANALYSIS REPORT")
        report.append(f"{'='*70}\n")

        # Primary emotion
        primary = emotional_state['primary_emotion']
        report.append(f"PRIMARY EMOTION: {primary['name'].upper()}")
        report.append(f"  Confidence: {primary['confidence']:.1%}")
        report.append(f"  Intensity: {primary['intensity']['description']}")
        report.append(f"  Description: {primary['description']}\n")

        # Risk assessment
        risk = emotional_state['risk_assessment']
        report.append(f"RISK ASSESSMENT:")
        report.append(f"  Level: {risk['level'].upper()}")
        report.append(f"  Priority: {risk['priority']}")
        report.append(
            f"  Requires Escalation: {'YES' if risk['requires_escalation'] else 'NO'}\n")

        # Emotional dimensions
        dimensions = emotional_state['emotional_dimensions']
        report.append(f"EMOTIONAL DIMENSIONS:")
        report.append(
            f"  Valence: {dimensions['valence']['category']} ({dimensions['valence']['value']:.2f})")
        report.append(
            f"  Arousal: {dimensions['arousal']['category']} ({dimensions['arousal']['value']:.2f})\n")

        # Secondary emotions
        if emotional_state['secondary_emotions']:
            report.append(f"SECONDARY EMOTIONS:")
            for sec in emotional_state['secondary_emotions']:
                report.append(f"  - {sec['name']}: {sec['probability']:.1%}")
            report.append("")

        # Stability
        stability = emotional_state['stability']
        report.append(f"EMOTIONAL STABILITY:")
        report.append(f"  Score: {stability['score']:.2f}")
        report.append(
            f"  Status: {'Stable' if stability['is_stable'] else 'Mixed/Unstable'}\n")

        # Audio indicators
        if 'audio_indicators' in emotional_state and emotional_state['audio_indicators']:
            report.append(f"AUDIO INDICATORS:")
            for indicator in emotional_state['audio_indicators']:
                report.append(
                    f"  • {indicator['feature']}: {indicator['value']}")
                report.append(f"    → {indicator['interpretation']}")
            report.append("")

        # Transcription indicators
        if 'transcription_indicators' in emotional_state and emotional_state['transcription_indicators']:
            report.append(f"TRANSCRIPTION INDICATORS:")
            for indicator in emotional_state['transcription_indicators']:
                report.append(
                    f"  • {indicator['feature']}: {indicator['value']}")
                report.append(f"    → {indicator['interpretation']}")
            report.append("")

        report.append(f"{'='*70}")

        return '\n'.join(report)

    def save_emotional_state(self, emotional_state, output_dir='emotional_states'):
        """Save emotional state analysis to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        emotion = emotional_state['primary_emotion']['emotion']

        # Save JSON
        json_file = output_path / f"emotion_{emotion}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(emotional_state, f, indent=2)

        # Save report
        report_file = output_path / f"emotion_{emotion}_{timestamp}_report.txt"
        report = self.generate_emotion_report(emotional_state)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Emotional state saved:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}\n")

        return {
            'json': str(json_file),
            'report': str(report_file)
        }

    def visualize_emotional_state(self, emotional_state, output_dir='emotional_visualizations'):
        """Create visualizations of emotional state"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return {}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_plots = {}

        # 1. Emotion Probabilities Bar Chart
        plt.figure(figsize=(10, 6))
        probs = emotional_state['all_emotion_probabilities']
        emotions = list(probs.keys())
        probabilities = list(probs.values())

        colors = ['red' if e == emotional_state['primary_emotion']['emotion'] else 'skyblue'
                  for e in emotions]

        plt.bar(emotions, probabilities, color=colors, alpha=0.7)
        plt.xlabel('Emotions')
        plt.ylabel('Probability')
        plt.title('Emotion Classification Probabilities')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        prob_file = output_path / f'emotion_probabilities_{timestamp}.png'
        plt.savefig(prob_file, dpi=150)
        plt.close()
        saved_plots['probabilities'] = str(prob_file)

        # 2. Valence-Arousal Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        valence = emotional_state['emotional_dimensions']['valence']['value']
        arousal = emotional_state['emotional_dimensions']['arousal']['value']
        emotion_name = emotional_state['primary_emotion']['name']

        # Create quadrant plot
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Plot point
        ax.scatter([valence], [arousal], s=500, c='red',
                   alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(emotion_name, (valence, arousal), fontsize=12, fontweight='bold',
                    xytext=(10, 10), textcoords='offset points')

        # Labels
        ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=12)
        ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=12)
        ax.set_title('Emotional State: Valence-Arousal Model',
                     fontsize=14, fontweight='bold')

        # Quadrant labels
        ax.text(-0.8, 0.9, 'Angry/Frustrated\n(Negative, High)',
                ha='center', fontsize=10, alpha=0.5)
        ax.text(0.8, 0.9, 'Happy/Excited\n(Positive, High)',
                ha='center', fontsize=10, alpha=0.5)
        ax.text(-0.8, 0.1, 'Sad/Depressed\n(Negative, Low)',
                ha='center', fontsize=10, alpha=0.5)
        ax.text(0.8, 0.1, 'Calm/Satisfied\n(Positive, Low)',
                ha='center', fontsize=10, alpha=0.5)

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        va_file = output_path / f'valence_arousal_{timestamp}.png'
        plt.savefig(va_file, dpi=150)
        plt.close()
        saved_plots['valence_arousal'] = str(va_file)

        print(f"✓ Visualizations saved to: {output_dir}/\n")

        return saved_plots


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='CSR Call Recording - Emotional State Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--prediction',
        help='Path to ML classifier prediction JSON file',
        required=True
    )

    parser.add_argument(
        '--features',
        help='Path to MFCC features JSON file (optional)'
    )

    parser.add_argument(
        '--transcription',
        help='Path to Whisper transcription JSON file (optional)'
    )

    parser.add_argument(
        '--output',
        default='emotional_states',
        help='Output directory for emotional state files'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )

    args = parser.parse_args()

    # Load prediction
    with open(args.prediction, 'r') as f:
        prediction = json.load(f)

    # Load optional features
    audio_features = None
    if args.features:
        with open(args.features, 'r') as f:
            audio_features = json.load(f)

    # Load optional transcription
    transcription = None
    if args.transcription:
        with open(args.transcription, 'r') as f:
            transcription = json.load(f)

    # Classify emotional state
    classifier = EmotionalStateClassifier()
    emotional_state = classifier.classify_emotional_state(
        prediction,
        audio_features=audio_features,
        transcription=transcription
    )

    # Print report
    report = classifier.generate_emotion_report(emotional_state)
    print(report)

    # Save results
    classifier.save_emotional_state(emotional_state, output_dir=args.output)

    # Visualize if requested
    if args.visualize:
        classifier.visualize_emotional_state(emotional_state)


if __name__ == '__main__':
    main()
