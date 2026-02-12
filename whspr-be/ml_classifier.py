"""
CSR Call Recording - ML Classifier Module
Emotion classification using SVM, Random Forest, and KNN
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization disabled.")

warnings.filterwarnings('ignore')


class EmotionClassifier:
    """
    ML Classifier for Emotional State Classification
    Supports: SVM, Random Forest, KNN
    """

    # Emotion labels
    EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'satisfied']

    # Available classifiers
    CLASSIFIERS = {
        'svm': 'Support Vector Machine',
        'rf': 'Random Forest',
        'knn': 'K-Nearest Neighbors'
    }

    def __init__(self, classifier_type='svm', random_state=42):
        """
        Initialize Emotion Classifier

        Args:
            classifier_type (str): 'svm', 'rf', or 'knn'
            random_state (int): Random seed for reproducibility
        """
        self.classifier_type = classifier_type.lower()
        self.random_state = random_state

        if self.classifier_type not in self.CLASSIFIERS:
            raise ValueError(
                f"Classifier must be one of: {list(self.CLASSIFIERS.keys())}")

        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False

        # Training history
        self.training_history = {
            'classifier_type': classifier_type,
            'trained_at': None,
            'training_samples': 0,
            'test_accuracy': 0.0,
            'feature_count': 0,
            'emotions': []
        }

        print(f"{'='*70}")
        print(f"ML Emotion Classifier Initialized")
        print(f"{'='*70}")
        print(f"Classifier: {self.CLASSIFIERS[self.classifier_type]}")
        print(f"Random State: {random_state}")
        print(f"{'='*70}\n")

    def _create_classifier(self, **kwargs):
        """
        Create classifier instance based on type

        Args:
            **kwargs: Classifier-specific parameters

        Returns:
            Classifier instance
        """
        if self.classifier_type == 'svm':
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=self.random_state
            )

        elif self.classifier_type == 'rf':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=self.random_state
            )

        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'uniform'),
                metric=kwargs.get('metric', 'minkowski')
            )

    def train(self, X, y, test_size=0.2, **classifier_params):
        """
        Train the classifier

        Args:
            X (np.array): Feature matrix (n_samples, n_features)
            y (np.array): Labels (n_samples,)
            test_size (float): Proportion of test set
            **classifier_params: Classifier-specific parameters

        Returns:
            dict: Training results
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {self.CLASSIFIERS[self.classifier_type].upper()}")
        print(f"{'='*70}\n")

        # Validate input
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X:{len(X)}, y:{len(y)}")

        print(f"[Data] Total samples: {len(X)}")
        print(f"[Data] Features per sample: {X.shape[1]}")
        print(f"[Data] Unique emotions: {np.unique(y)}")
        print(f"[Data] Test size: {test_size * 100}%\n")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        print(f"[Split] Training samples: {len(X_train)}")
        print(f"[Split] Test samples: {len(X_test)}\n")

        # Scale features
        print("[Preprocessing] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✓ Features scaled\n")

        # Create and train classifier
        print(
            f"[Training] Training {self.CLASSIFIERS[self.classifier_type]}...")
        self.model = self._create_classifier(**classifier_params)
        self.model.fit(X_train_scaled, y_train)
        print("✓ Training complete\n")

        # Evaluate on test set
        print("[Evaluation] Testing model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Cross-validation
        print("[Cross-Validation] Running 5-fold CV...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )

        # Update training history
        self.is_trained = True
        self.training_history.update({
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_count': X.shape[1],
            'emotions': self.label_encoder.classes_.tolist(),
            'confusion_matrix': cm.tolist()
        })

        # Display results
        print(f"\n{'='*70}")
        print(f"TRAINING RESULTS")
        print(f"{'='*70}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nCross-Validation (5-fold):")
        print(
            f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  CV Scores: {cv_scores}")
        print(f"{'='*70}\n")

        # Detailed classification report
        print("Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'predictions': y_pred,
            'true_labels': y_test,
            'predicted_probabilities': y_pred_proba
        }

        return results

    def predict(self, X, return_probabilities=True):
        """
        Predict emotions for new samples

        Args:
            X (np.array): Feature matrix (n_samples, n_features)
            return_probabilities (bool): Return probability scores

        Returns:
            dict: Predictions with labels and probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)

        results = {
            'predictions': y_pred_labels.tolist(),
            'num_samples': len(X)
        }

        if return_probabilities:
            y_pred_proba = self.model.predict_proba(X_scaled)

            # Get probabilities for each emotion
            probabilities = []
            for i, probs in enumerate(y_pred_proba):
                prob_dict = {
                    emotion: float(prob)
                    for emotion, prob in zip(self.label_encoder.classes_, probs)
                }
                # Sort by probability
                prob_dict = dict(
                    sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                probabilities.append({
                    'predicted_emotion': y_pred_labels[i],
                    'confidence': float(max(probs)),
                    'all_probabilities': prob_dict
                })

            results['probabilities'] = probabilities

        return results

    def predict_single(self, feature_vector):
        """
        Predict emotion for a single call recording

        Args:
            feature_vector (np.array or list): Single feature vector

        Returns:
            dict: Prediction result
        """
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)

        result = self.predict(feature_vector.reshape(1, -1))

        prediction = {
            'emotion': result['predictions'][0],
            'confidence': result['probabilities'][0]['confidence'],
            'all_probabilities': result['probabilities'][0]['all_probabilities']
        }

        return prediction

    def optimize_hyperparameters(self, X, y, cv=5):
        """
        Optimize classifier hyperparameters using GridSearchCV

        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            cv (int): Number of cross-validation folds

        Returns:
            dict: Best parameters and scores
        """
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*70}\n")

        # Encode labels and scale features
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grids
        param_grids = {
            'svm': {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'rf': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }

        param_grid = param_grids[self.classifier_type]

        print(f"[Optimization] Searching parameter space...")
        print(f"[Optimization] Parameter grid: {param_grid}\n")

        # Create base classifier
        base_model = self._create_classifier()

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_scaled, y_encoded)

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"{'='*70}\n")

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True

        return {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_
        }

    def save_model(self, filepath='emotion_classifier_model.pkl'):
        """
        Save trained model to file

        Args:
            filepath (str): Path to save model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classifier_type': self.classifier_type,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to: {filepath}")
        return filepath

    def load_model(self, filepath):
        """
        Load trained model from file

        Args:
            filepath (str): Path to saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.classifier_type = model_data['classifier_type']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']

        print(f"✓ Model loaded from: {filepath}")
        print(f"  Classifier: {self.CLASSIFIERS[self.classifier_type]}")
        print(f"  Trained at: {self.training_history['trained_at']}")
        print(
            f"  Training accuracy: {self.training_history['test_accuracy']:.4f}")

    def visualize_results(self, results, output_dir='visualizations'):
        """
        Create visualizations of training results

        Args:
            results (dict): Training results from train()
            output_dir (str): Directory to save plots

        Returns:
            dict: Paths to saved visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return {}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_plots = {}

        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(
            f'Confusion Matrix - {self.CLASSIFIERS[self.classifier_type]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = output_path / f'{self.classifier_type}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=150)
        plt.close()
        saved_plots['confusion_matrix'] = str(cm_path)

        # 2. Classification Report Heatmap
        report_dict = results['classification_report']
        report_data = []
        emotions = [e for e in report_dict.keys() if e not in [
            'accuracy', 'macro avg', 'weighted avg']]

        for emotion in emotions:
            report_data.append([
                report_dict[emotion]['precision'],
                report_dict[emotion]['recall'],
                report_dict[emotion]['f1-score']
            ])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            report_data,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=emotions
        )
        plt.title(
            f'Classification Metrics - {self.CLASSIFIERS[self.classifier_type]}')
        plt.tight_layout()
        metrics_path = output_path / f'{self.classifier_type}_metrics.png'
        plt.savefig(metrics_path, dpi=150)
        plt.close()
        saved_plots['metrics'] = str(metrics_path)

        # 3. Cross-Validation Scores
        plt.figure(figsize=(10, 6))
        cv_scores = results['cv_scores']
        folds = range(1, len(cv_scores) + 1)
        plt.bar(folds, cv_scores, color='skyblue', alpha=0.7)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--',
                    label=f'Mean: {cv_scores.mean():.4f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title(
            f'Cross-Validation Scores - {self.CLASSIFIERS[self.classifier_type]}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        cv_path = output_path / f'{self.classifier_type}_cv_scores.png'
        plt.savefig(cv_path, dpi=150)
        plt.close()
        saved_plots['cv_scores'] = str(cv_path)

        # 4. Feature Importance (Random Forest only)
        if self.classifier_type == 'rf' and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(indices)), importances[indices])
            plt.title('Top 20 Feature Importances - Random Forest')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            fi_path = output_path / 'rf_feature_importance.png'
            plt.savefig(fi_path, dpi=150)
            plt.close()
            saved_plots['feature_importance'] = str(fi_path)

        print(f"✓ Visualizations saved to: {output_dir}/\n")

        return saved_plots

    def compare_classifiers(self, X, y, test_size=0.2):
        """
        Compare all three classifiers on the same dataset

        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            test_size (float): Test set proportion

        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*70}")
        print(f"COMPARING ALL CLASSIFIERS")
        print(f"{'='*70}\n")

        results = {}

        for clf_type in ['svm', 'rf', 'knn']:
            print(f"\nTraining {self.CLASSIFIERS[clf_type]}...")
            print(f"{'-'*70}")

            classifier = EmotionClassifier(
                classifier_type=clf_type, random_state=self.random_state)
            clf_results = classifier.train(X, y, test_size=test_size)

            results[clf_type] = {
                'accuracy': clf_results['accuracy'],
                'precision': clf_results['precision'],
                'recall': clf_results['recall'],
                'f1_score': clf_results['f1_score'],
                'cv_mean': clf_results['cv_mean'],
                'cv_std': clf_results['cv_std']
            }

        # Display comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*70}\n")

        comparison_df = pd.DataFrame(results).T
        comparison_df.index = [self.CLASSIFIERS[idx]
                               for idx in comparison_df.index]
        print(comparison_df.to_string())
        print(f"\n{'='*70}\n")

        # Find best classifier
        best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best Classifier: {self.CLASSIFIERS[best_clf[0]]}")
        print(f"Accuracy: {best_clf[1]['accuracy']:.4f}\n")

        return results


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='CSR Call Recording - ML Emotion Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SVM classifier
  python ml_classifier.py --features features.npy --labels labels.npy --classifier svm
  
  # Train and compare all classifiers
  python ml_classifier.py --features features.npy --labels labels.npy --compare
  
  # Train with optimization
  python ml_classifier.py --features features.npy --labels labels.npy --classifier rf --optimize
  
  # Predict emotions
  python ml_classifier.py --predict feature_vector.npy --model saved_model.pkl
        """
    )

    parser.add_argument(
        '--features',
        help='Path to feature matrix (.npy file)'
    )

    parser.add_argument(
        '--labels',
        help='Path to labels (.npy file)'
    )

    parser.add_argument(
        '--classifier',
        choices=['svm', 'rf', 'knn'],
        default='svm',
        help='Classifier type (default: svm)'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize hyperparameters'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all classifiers'
    )

    parser.add_argument(
        '--save-model',
        help='Path to save trained model'
    )

    parser.add_argument(
        '--predict',
        help='Path to feature vector for prediction'
    )

    parser.add_argument(
        '--model',
        help='Path to saved model for prediction'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization plots'
    )

    args = parser.parse_args()

    # Prediction mode
    if args.predict:
        if not args.model:
            print("Error: --model required for prediction")
            sys.exit(1)

        classifier = EmotionClassifier()
        classifier.load_model(args.model)

        features = np.load(args.predict)
        result = classifier.predict(features)

        print(f"\n{'='*70}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*70}")
        for i, pred in enumerate(result['probabilities']):
            print(f"\nSample {i+1}:")
            print(f"  Emotion: {pred['predicted_emotion']}")
            print(f"  Confidence: {pred['confidence']:.4f}")
            print(f"  All probabilities:")
            for emotion, prob in pred['all_probabilities'].items():
                print(f"    {emotion}: {prob:.4f}")
        print(f"{'='*70}\n")

        sys.exit(0)

    # Training mode
    if not args.features or not args.labels:
        print("Error: --features and --labels required for training")
        sys.exit(1)

    # Load data
    X = np.load(args.features)
    y = np.load(args.labels)

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")

    # Compare mode
    if args.compare:
        classifier = EmotionClassifier()
        results = classifier.compare_classifiers(
            X, y, test_size=args.test_size)
        sys.exit(0)

    # Single classifier mode
    classifier = EmotionClassifier(classifier_type=args.classifier)

    if args.optimize:
        classifier.optimize_hyperparameters(X, y)
        results = classifier.train(X, y, test_size=args.test_size)
    else:
        results = classifier.train(X, y, test_size=args.test_size)

    # Visualize
    if args.visualize:
        classifier.visualize_results(results)

    # Save model
    if args.save_model:
        classifier.save_model(args.save_model)
    else:
        # Default save path
        default_path = f'{args.classifier}_emotion_classifier.pkl'
        classifier.save_model(default_path)

    print(f"\n✓ Training complete!")


if __name__ == '__main__':
    main()
