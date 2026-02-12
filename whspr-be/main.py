"""
CSR Call Recording Analysis API
FastAPI backend integrating Whisper ASR, MFCC, ML Classification, and Recommendations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import tempfile
import uuid

# Import CSR analysis modules
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from whisper_asr_module import CSRCallTranscriber
    from mfcc_feature_extraction import MFCCFeatureExtractor
    from ml_classifier import EmotionClassifier
    from emotional_state_classifier import EmotionalStateClassifier
    from csr_emotion_recommendations import CSREmotionClassifier
except ImportError as e:
    print(f"Warning: Could not import CSR modules: {e}")
    print("Make sure all module files are in the same directory")


# Initialize FastAPI
app = FastAPI(
    title="CSR Call Recording Analysis API",
    version="1.0.0",
    description="AI-powered emotional analysis of customer service call recordings"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    # Update with your Next.js URL
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Global instances (initialize once)
whisper_transcriber = None
mfcc_extractor = None
ml_classifier = None
emotion_classifier = None
recommendation_engine = None

# Processing status storage (in production, use Redis or database)
processing_status = {}


def initialize_models():
    """Initialize all models on startup"""
    global whisper_transcriber, mfcc_extractor, ml_classifier, emotion_classifier, recommendation_engine

    print("Initializing models...")

    try:
        # Initialize Whisper (use 'base' for balance of speed/accuracy)
        whisper_transcriber = CSRCallTranscriber(model_size='base')
        print("✓ Whisper ASR initialized")

        # Initialize MFCC extractor
        mfcc_extractor = MFCCFeatureExtractor()
        print("✓ MFCC Feature Extractor initialized")

        # Initialize ML Classifier (load pre-trained model if exists)
        ml_classifier = EmotionClassifier(classifier_type='svm')
        model_path = MODELS_DIR / "trained_emotion_classifier.pkl"
        if model_path.exists():
            ml_classifier.load_model(str(model_path))
            print("✓ ML Classifier loaded from file")
        else:
            print("⚠ Warning: No pre-trained model found. You need to train first!")
            print(f"  Expected path: {model_path}")

        # Initialize Emotional State Classifier
        emotion_classifier = EmotionalStateClassifier()
        print("✓ Emotional State Classifier initialized")

        # Initialize Recommendation Engine
        recommendation_engine = CSREmotionClassifier()
        print("✓ Recommendation Engine initialized")

        print("✓ All models initialized successfully!\n")

    except Exception as e:
        print(f"Error initializing models: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("\n" + "="*70)
    print("CSR CALL RECORDING ANALYSIS API")
    print("="*70 + "\n")
    initialize_models()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CSR Call Recording Analysis API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/api/upload (POST)",
            "status": "/api/status/{job_id} (GET)",
            "results": "/api/results/{job_id} (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_ready = all([
        whisper_transcriber is not None,
        mfcc_extractor is not None,
        ml_classifier is not None,
        emotion_classifier is not None,
        recommendation_engine is not None
    ])

    return {
        "status": "healthy" if models_ready else "initializing",
        "models_ready": models_ready,
        "timestamp": datetime.now().isoformat()
    }


async def process_recording_pipeline(job_id: str, audio_path: str):
    """
    Complete processing pipeline for CSR call recording

    Pipeline:
    1. Whisper ASR (Transcription)
    2. MFCC Feature Extraction
    3. ML Classifier (SVM/RF/KNN)
    4. Emotional State Classification
    5. Recommendation Engine
    """

    try:
        processing_status[job_id] = {
            "status": "processing",
            "stage": "started",
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }

        # Create job-specific results directory
        job_results_dir = RESULTS_DIR / job_id
        job_results_dir.mkdir(exist_ok=True)

        results = {
            "job_id": job_id,
            "audio_file": Path(audio_path).name,
            "started_at": datetime.now().isoformat()
        }

        # STAGE 1: Whisper ASR Transcription
        print(f"\n[Job {job_id}] Stage 1: Transcription")
        processing_status[job_id].update({
            "stage": "transcription",
            "progress": 20
        })

        transcription = whisper_transcriber.transcribe_call(audio_path)
        results["transcription"] = {
            "full_text": transcription["full_text"],
            "duration": transcription["duration_formatted"],
            "word_count": transcription["word_count"],
            "segment_count": transcription["segment_count"]
        }

        # Save transcription
        transcription_file = job_results_dir / "transcription.json"
        with open(transcription_file, 'w') as f:
            json.dump(transcription, f, indent=2)

        print(f"✓ Transcription complete: {transcription['word_count']} words")

        # STAGE 2: MFCC Feature Extraction
        print(f"\n[Job {job_id}] Stage 2: Feature Extraction")
        processing_status[job_id].update({
            "stage": "feature_extraction",
            "progress": 40
        })

        features = mfcc_extractor.extract_all_features(audio_path)
        results["features"] = {
            "feature_vector_length": features["feature_vector_length"],
            "duration_seconds": features["duration_seconds"]
        }

        # Save features
        features_file = job_results_dir / "features.json"
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)

        print(
            f"✓ Feature extraction complete: {features['feature_vector_length']} features")

        # STAGE 3: ML Classification
        print(f"\n[Job {job_id}] Stage 3: Emotion Classification")
        processing_status[job_id].update({
            "stage": "classification",
            "progress": 60
        })

        # Check if model is trained
        if not ml_classifier.is_trained:
            raise Exception(
                "ML Classifier not trained. Please train the model first.")

        # Predict emotion
        import numpy as np
        feature_vector = np.array(features["feature_vector"])
        prediction = ml_classifier.predict_single(feature_vector)

        results["emotion_prediction"] = prediction

        print(
            f"✓ Emotion classified: {prediction['emotion']} ({prediction['confidence']:.2%})")

        # STAGE 4: Emotional State Analysis
        print(f"\n[Job {job_id}] Stage 4: Emotional State Analysis")
        processing_status[job_id].update({
            "stage": "emotional_analysis",
            "progress": 80
        })

        emotional_state = emotion_classifier.classify_emotional_state(
            prediction,
            audio_features=features,
            transcription=transcription
        )

        results["emotional_state"] = {
            "primary_emotion": emotional_state["primary_emotion"],
            "risk_assessment": emotional_state["risk_assessment"],
            "emotional_dimensions": emotional_state["emotional_dimensions"],
            "stability": emotional_state["stability"]
        }

        # Save emotional state
        emotional_state_file = job_results_dir / "emotional_state.json"
        with open(emotional_state_file, 'w') as f:
            json.dump(emotional_state, f, indent=2)

        print(
            f"✓ Emotional state analyzed: {emotional_state['primary_emotion']['name']}")

        # STAGE 5: Generate Recommendations
        print(f"\n[Job {job_id}] Stage 5: Generating Recommendations")
        processing_status[job_id].update({
            "stage": "recommendations",
            "progress": 90
        })

        # Use CSR-specific classifier for recommendations
        csr_emotional_state = recommendation_engine.classify_emotional_state(
            prediction)
        recommendation = recommendation_engine.generate_recommendation(
            csr_emotional_state)

        results["recommendation"] = recommendation
        results["csr_emotional_state"] = {
            "affective_load": csr_emotional_state["affective_load_category"],
            "confidence_level": csr_emotional_state["confidence_level"],
            "risk_level": csr_emotional_state["psychological_indicators"]["risk_level"]
        }

        # Generate report
        report = recommendation_engine.generate_report(
            csr_emotional_state, recommendation)
        report_file = job_results_dir / "recommendation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✓ Recommendations generated: {recommendation['priority']}")

        # FINAL: Save complete results
        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        complete_results_file = job_results_dir / "complete_results.json"
        with open(complete_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        processing_status[job_id].update({
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "results": results
        })

        print(f"\n✓ Pipeline completed for job {job_id}")
        print(f"Results saved to: {job_results_dir}")

    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error in pipeline for job {job_id}: {error_msg}")

        processing_status[job_id].update({
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.now().isoformat()
        })


@app.post("/api/upload")
async def upload_recording(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process CSR call recording

    Accepts: audio files (mp3, wav, m4a, ogg, webm)
    Returns: job_id for tracking processing status
    """

    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{job_id}_{timestamp}{file_ext}"
    file_path = UPLOAD_DIR / filename

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n[Upload] File saved: {filename}")
        print(f"[Upload] Job ID: {job_id}")
        print(
            f"[Upload] File size: {file_path.stat().st_size / (1024*1024):.2f} MB")

        # Initialize processing status
        processing_status[job_id] = {
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat()
        }

        # Start background processing
        background_tasks.add_task(
            process_recording_pipeline,
            job_id,
            str(file_path)
        )

        return {
            "success": True,
            "job_id": job_id,
            "filename": file.filename,
            "message": "Recording uploaded successfully. Processing started.",
            "status_url": f"/api/status/{job_id}",
            "results_url": f"/api/results/{job_id}"
        }

    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status for a job"""

    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    return processing_status[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get complete results for a completed job"""

    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    status = processing_status[job_id]

    if status["status"] != "completed":
        return {
            "status": status["status"],
            "message": f"Job is {status['status']}. Results not yet available.",
            "current_stage": status.get("stage"),
            "progress": status.get("progress", 0)
        }

    # Load complete results
    results_file = RESULTS_DIR / job_id / "complete_results.json"

    if not results_file.exists():
        raise HTTPException(
            status_code=500,
            detail="Results file not found"
        )

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


@app.get("/api/jobs")
async def list_jobs():
    """List all processing jobs"""

    jobs = []
    for job_id, status in processing_status.items():
        jobs.append({
            "job_id": job_id,
            "status": status["status"],
            "filename": status.get("filename"),
            "uploaded_at": status.get("uploaded_at"),
            "progress": status.get("progress", 0)
        })

    return {
        "total_jobs": len(jobs),
        "jobs": jobs
    }


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""

    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="Job ID not found"
        )

    try:
        # Delete results directory
        results_dir = RESULTS_DIR / job_id
        if results_dir.exists():
            shutil.rmtree(results_dir)

        # Delete uploaded file (find it by job_id prefix)
        for file in UPLOAD_DIR.glob(f"{job_id}_*"):
            file.unlink()

        # Remove from status
        del processing_status[job_id]

        return {
            "success": True,
            "message": f"Job {job_id} deleted successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting job: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting CSR Call Recording Analysis API Server")
    print("="*70 + "\n")

    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
