"""
ml_classifier.py
CSR Call Recording - ML Classifier Module
Emotion classification using SVM, Random Forest, and KNN
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
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
)

warnings.filterwarnings("ignore")


class EmotionClassifier:
    EMOTIONS = ["angry", "happy", "sad", "neutral", "frustrated", "satisfied"]

    CLASSIFIERS = {
        "svm": "Support Vector Machine",
        "rf": "Random Forest",
        "knn": "K-Nearest Neighbors",
    }

    def __init__(self, classifier_type="svm", random_state=42):
        self.classifier_type = classifier_type.lower()
        self.random_state = random_state

        if self.classifier_type not in self.CLASSIFIERS:
            raise ValueError(
                f"Classifier must be one of: {list(self.CLASSIFIERS.keys())}"
            )

        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        # Multi-model holders
        self.svm_model = None
        self.rf_model = None
        self.knn_model = None
        self.svm_label_encoder = None
        self.rf_label_encoder = None
        self.knn_label_encoder = None

        self.training_history = {
            "classifier_type": self.classifier_type,
            "trained_at": None,
            "training_samples": 0,
            "test_accuracy": 0.0,
            "feature_count": 0,
            "emotions": [],
        }

        print("=" * 70)
        print("ML Emotion Classifier Initialized")
        print("=" * 70)
        print(f"Classifier: {self.CLASSIFIERS[self.classifier_type]}")
        print("=" * 70)

    def _create_classifier(self, classifier_type=None, **kwargs):
        clf_type = classifier_type or self.classifier_type

        if clf_type == "svm":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel=kwargs.get("kernel", "rbf"),
                    C=kwargs.get("C", 10),
                    gamma=kwargs.get("gamma", "scale"),
                    probability=True,
                    class_weight="balanced",
                    random_state=self.random_state,
                )),
            ])

        if clf_type == "rf":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                min_samples_leaf=kwargs.get("min_samples_leaf", 1),
                class_weight="balanced",
                random_state=self.random_state,
            )

        if clf_type == "knn":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(
                    n_neighbors=kwargs.get("n_neighbors", 5),
                    weights=kwargs.get("weights", "distance"),
                    metric=kwargs.get("metric", "minkowski"),
                )),
            ])

        raise ValueError(f"Unknown classifier type: {clf_type}")

    def train(self, X, y, test_size=0.2, **classifier_params):
        print("=" * 70)
        print(f"TRAINING {self.CLASSIFIERS[self.classifier_type].upper()}")
        print("=" * 70)

        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        self.model = self._create_classifier(**classifier_params)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred)
        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        self.is_trained = True
        self.training_history.update({
            "classifier_type": self.classifier_type,
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "feature_count": X.shape[1],
            "emotions": self.label_encoder.classes_.tolist(),
            "confusion_matrix": cm.tolist(),
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"CV mean: {cv_scores.mean():.4f}")

        print(classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        ))

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
            "predictions": y_pred.tolist(),
            "true_labels": y_test.tolist(),
            "predicted_probabilities": y_pred_proba.tolist(),
        }

    def train_all_models(self, X, y, test_size=0.2, save_dir="models"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = {}

        for clf_type in ["svm", "rf", "knn"]:
            print("\n" + "=" * 70)
            print(f"TRAINING {clf_type.upper()}")
            print("=" * 70)

            clf = EmotionClassifier(
                classifier_type=clf_type,
                random_state=self.random_state,
            )

            result = clf.train(X, y, test_size=test_size)
            save_path = Path(save_dir) / f"{clf_type}_emotion_model.pkl"
            clf.save_model(str(save_path))

            results[clf_type] = result

        return results

    def predict(self, X, return_probabilities=True):
        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "Model is not trained or loaded. Train model or load_model() first."
            )

        X = np.asarray(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        y_pred_encoded = self.model.predict(X)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)

        results = {
            "predictions": y_pred_labels.tolist(),
            "num_samples": len(X),
        }

        if return_probabilities:
            y_pred_proba = self.model.predict_proba(X)

            probabilities = []
            for i, probs in enumerate(y_pred_proba):
                prob_dict = {
                    emotion: float(prob)
                    for emotion, prob in zip(self.label_encoder.classes_, probs)
                }

                prob_dict = dict(
                    sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                )

                probabilities.append({
                    "predicted_emotion": y_pred_labels[i],
                    "confidence": float(max(probs)),
                    "all_probabilities": prob_dict,
                })

            results["probabilities"] = probabilities

        return results

    def predict_single(self, feature_vector):
        try:
            feature_vector = np.asarray(feature_vector)

            if len(feature_vector.shape) > 1:
                feature_vector = feature_vector.flatten()

            result = self.predict(feature_vector.reshape(1, -1))

            return {
                "emotion": result["predictions"][0],
                "confidence": result["probabilities"][0]["confidence"],
                "all_probabilities": result["probabilities"][0]["all_probabilities"],
            }

        except Exception as e:
            print(f"Error in predict_single: {e}")
            return {
                "error": str(e),
                "emotion": "neutral",
                "confidence": 0.0,
                "all_probabilities": {emotion: 0.0 for emotion in self.EMOTIONS},
            }

    def save_model(self, filepath):
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "classifier_type": self.classifier_type,
            "training_history": self.training_history,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")
        return filepath

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]
        self.classifier_type = model_data["classifier_type"]
        self.training_history = model_data.get("training_history", {})
        self.is_trained = model_data.get("is_trained", True)

        print(f"Model loaded: {filepath}")
        print(f"Classifier: {self.classifier_type}")

    def load_all_models(self, model_dir="models"):
        model_dir = Path(model_dir)

        paths = {
            "svm": model_dir / "svm_emotion_model.pkl",
            "rf": model_dir / "rf_emotion_model.pkl",
            "knn": model_dir / "knn_emotion_model.pkl",
        }

        for key, path in paths.items():
            if not path.exists():
                print(f"Missing model: {path}")
                continue

            with open(path, "rb") as f:
                data = pickle.load(f)

            if key == "svm":
                self.svm_model = data["model"]
                self.svm_label_encoder = data["label_encoder"]
            elif key == "rf":
                self.rf_model = data["model"]
                self.rf_label_encoder = data["label_encoder"]
            elif key == "knn":
                self.knn_model = data["model"]
                self.knn_label_encoder = data["label_encoder"]

            print(f"Loaded {key.upper()} model from {path}")

        loaded = [
            name.upper()
            for name, model in [
                ("svm", self.svm_model),
                ("rf", self.rf_model),
                ("knn", self.knn_model),
            ]
            if model is not None
        ]

        print(f"Loaded models: {loaded}")

    def _predict_with_model(self, model, label_encoder, features):
        if model is None or label_encoder is None:
            return None

        try:
            X = np.asarray(features)

            if len(X.shape) > 1:
                X = X.flatten()

            X = X.reshape(1, -1)

            pred_encoded = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            emotion = label_encoder.inverse_transform([pred_encoded])[0]

            prob_dict = {
                emotion_label: float(prob)
                for emotion_label, prob in zip(label_encoder.classes_, probs)
            }

            prob_dict = dict(
                sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "emotion": str(emotion).lower(),
                "confidence": float(max(probs)),
                "all_probabilities": prob_dict,
            }

        except Exception as e:
            print(f"Prediction failed: {e}")
            return None

    def predict_svm(self, features):
        return self._predict_with_model(
            self.svm_model,
            self.svm_label_encoder,
            features,
        )

    def predict_rf(self, features):
        return self._predict_with_model(
            self.rf_model,
            self.rf_label_encoder,
            features,
        )

    def predict_knn(self, features):
        return self._predict_with_model(
            self.knn_model,
            self.knn_label_encoder,
            features,
        )

    def optimize_hyperparameters(self, X, y, cv=5):
        X = np.asarray(X)
        y = np.asarray(y)

        y_encoded = self.label_encoder.fit_transform(y)

        param_grids = {
            "svm": {
                "svm__kernel": ["rbf", "linear"],
                "svm__C": [1, 10, 100],
                "svm__gamma": ["scale", "auto"],
            },
            "rf": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "knn": {
                "knn__n_neighbors": [3, 5, 7, 9],
                "knn__weights": ["uniform", "distance"],
                "knn__metric": ["euclidean", "manhattan", "minkowski"],
            },
        }

        base_model = self._create_classifier()
        param_grid = param_grids[self.classifier_type]

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y_encoded)

        self.model = grid_search.best_estimator_
        self.is_trained = True

        return {
            "best_score": float(grid_search.best_score_),
            "best_params": grid_search.best_params_,
        }

    def compare_classifiers(self, X, y, test_size=0.2):
        results = {}

        for clf_type in ["svm", "rf", "knn"]:
            classifier = EmotionClassifier(
                classifier_type=clf_type,
                random_state=self.random_state,
            )
            clf_results = classifier.train(X, y, test_size=test_size)

            results[clf_type] = {
                "accuracy": clf_results["accuracy"],
                "precision": clf_results["precision"],
                "recall": clf_results["recall"],
                "f1_score": clf_results["f1_score"],
                "cv_mean": clf_results["cv_mean"],
                "cv_std": clf_results["cv_std"],
            }

        print(pd.DataFrame(results).T)

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CSR Call Recording - ML Emotion Classifier"
    )

    parser.add_argument("--features", help="Path to features .npy")
    parser.add_argument("--labels", help="Path to labels .npy")
    parser.add_argument(
        "--classifier",
        choices=["svm", "rf", "knn"],
        default="svm",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--save-model", help="Path to save trained model")
    parser.add_argument("--model", help="Path to saved model")
    parser.add_argument("--predict", help="Path to feature vector .npy")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--model-dir", default="models")

    args = parser.parse_args()

    if args.predict:
        if not args.model:
            print("Error: --model required for prediction")
            sys.exit(1)

        classifier = EmotionClassifier()
        classifier.load_model(args.model)

        features = np.load(args.predict)
        result = classifier.predict(features)

        print(result)
        sys.exit(0)

    if not args.features or not args.labels:
        print("Error: --features and --labels are required")
        sys.exit(1)

    X = np.load(args.features)
    y = np.load(args.labels)

    if args.train_all:
        classifier = EmotionClassifier()
        classifier.train_all_models(
            X,
            y,
            test_size=args.test_size,
            save_dir=args.model_dir,
        )
        print("All models trained and saved.")
        sys.exit(0)

    if args.compare:
        classifier = EmotionClassifier()
        classifier.compare_classifiers(X, y, test_size=args.test_size)
        sys.exit(0)

    classifier = EmotionClassifier(classifier_type=args.classifier)
    classifier.train(X, y, test_size=args.test_size)

    if args.save_model:
        classifier.save_model(args.save_model)
    else:
        classifier.save_model(
            f"{args.model_dir}/{args.classifier}_emotion_model.pkl"
        )


if __name__ == "__main__":
    main()