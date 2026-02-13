"""
CSR Call Recording Analysis API - Complete Fixed Version
FastAPI backend with emotion analysis and CSR recommendations
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import traceback
import json
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENSURE PROPER PYTHON PATH
# ============================================================================
CURRENT_DIR = Path(__file__).parent.absolute()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ============================================================================
# IMPORT ALL MODULES WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================

# Track which modules loaded successfully
modules_status = {
    'whisper': False,
    'mfcc': False,
    'emotion_classifier': False,
    'state_classifier': False,
    'recommendation_engine': False
}

# Import WhisperTranscriber
WhisperTranscriber = None
try:
    from whisper_asr_module import CSRCallTranscriber
    WhisperTranscriber = CSRCallTranscriber  # Assign the imported class
    modules_status['whisper'] = True
    logger.info("✓ WhisperTranscriber imported successfully")
except ImportError as e:
    logger.warning(f"✗ WhisperTranscriber import failed: {e}")
except Exception as e:
    logger.error(f"✗ Unexpected error importing WhisperTranscriber: {e}")

# Import MFCCFeatureExtractor
MFCCFeatureExtractor = None
try:
    from mfcc_feature_extraction import MFCCFeatureExtractor
    modules_status['mfcc'] = True
    logger.info("✓ MFCCFeatureExtractor imported successfully")
except ImportError as e:
    logger.warning(f"✗ MFCCFeatureExtractor import failed: {e}")
except Exception as e:
    logger.error(f"✗ Unexpected error importing MFCCFeatureExtractor: {e}")

# Import EmotionClassifier
EmotionClassifier = None
try:
    from ml_classifier import EmotionClassifier
    modules_status['emotion_classifier'] = True
    logger.info("✓ EmotionClassifier imported successfully")
except ImportError as e:
    logger.warning(f"✗ EmotionClassifier import failed: {e}")
except Exception as e:
    logger.error(f"✗ Unexpected error importing EmotionClassifier: {e}")

# Import EmotionalStateClassifier
EmotionalStateClassifier = None
try:
    from emotional_state_classifier import EmotionalStateClassifier
    modules_status['state_classifier'] = True
    logger.info("✓ EmotionalStateClassifier imported successfully")
except ImportError as e:
    logger.warning(f"✗ EmotionalStateClassifier import failed: {e}")
except Exception as e:
    logger.error(f"✗ Unexpected error importing EmotionalStateClassifier: {e}")

# Import CSREmotionClassifier - WITH SPECIAL HANDLING
CSREmotionClassifier = None
try:
    # First check if file exists
    csr_file = CURRENT_DIR / 'csr_emotion_recommendations.py'
    if not csr_file.exists():
        logger.error(f"✗ File not found: {csr_file}")
    else:
        # Try to compile first to catch syntax errors
        try:
            with open(csr_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(csr_file), 'exec')
            logger.info("✓ csr_emotion_recommendations.py syntax check passed")
        except SyntaxError as se:
            logger.error(f"✗ Syntax error in csr_emotion_recommendations.py at line {se.lineno}: {se.msg}")
        
        # Now try to import
        from csr_emotion_recommendations import CSREmotionClassifier
        modules_status['recommendation_engine'] = True
        logger.info("✓ CSREmotionClassifier imported successfully")
        
except ImportError as e:
    logger.warning(f"✗ CSREmotionClassifier import failed: {e}")
    logger.warning(f"   Make sure csr_emotion_recommendations.py is in: {CURRENT_DIR}")
except SyntaxError as e:
    logger.error(f"✗ Syntax error in csr_emotion_recommendations.py: {e}")
except Exception as e:
    logger.error(f"✗ Unexpected error importing CSREmotionClassifier: {e}")
    logger.error(traceback.format_exc())

# ============================================================================
# CREATE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="CSR Call Recording Analysis API",
    description="Emotion detection and CSR recommendation system for call recordings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CONFIGURE CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL MODEL INSTANCES
# ============================================================================

transcriber = None
feature_extractor = None
emotion_classifier = None
state_classifier = None
recommendation_engine = None

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

# Create all necessary directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, OUTPUT_DIR / "recommendations"]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Directory ready: {directory}")

# ============================================================================
# MODEL INITIALIZATION FUNCTION
# ============================================================================

def initialize_models():
    """Initialize all ML models and components with comprehensive error handling"""
    global transcriber, feature_extractor, emotion_classifier
    global state_classifier, recommendation_engine
    
    logger.info("="*70)
    logger.info("CSR CALL RECORDING ANALYSIS API - INITIALIZATION")
    logger.info("="*70)
    
    # Initialize Whisper ASR
    if modules_status.get('whisper') and WhisperTranscriber:
        try:
            transcriber = WhisperTranscriber(model_size='base', language='en')
            logger.info("✓ Whisper ASR initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Whisper ASR: {e}")
            transcriber = None
    else:
        logger.warning("⚠ Whisper ASR module not available - transcription disabled")
    
    # Initialize MFCC Feature Extractor
    if modules_status.get('mfcc') and MFCCFeatureExtractor:
        try:
            feature_extractor = MFCCFeatureExtractor()
            logger.info("✓ MFCC Feature Extractor initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize MFCC Feature Extractor: {e}")
            feature_extractor = None
    else:
        logger.warning("⚠ MFCC Feature Extractor module not available")
    
    # Initialize ML Emotion Classifier
    if modules_status.get('emotion_classifier') and EmotionClassifier:
        try:
            emotion_classifier = EmotionClassifier()
            logger.info("✓ ML Emotion Classifier initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Emotion Classifier: {e}")
            emotion_classifier = None
    else:
        logger.warning("⚠ Emotion Classifier module not available")
    
    # Initialize Emotional State Classifier
    if modules_status.get('state_classifier') and EmotionalStateClassifier:
        try:
            state_classifier = EmotionalStateClassifier()
            logger.info("✓ Emotional State Classifier initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize State Classifier: {e}")
            state_classifier = None
    else:
        logger.warning("⚠ Emotional State Classifier module not available")
    
    # Initialize CSR Recommendation Engine
    if modules_status.get('recommendation_engine') and CSREmotionClassifier:
        try:
            recommendation_engine = CSREmotionClassifier()
            logger.info("✓ CSR Recommendation Engine initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize CSR Recommendation Engine: {e}")
            logger.error(traceback.format_exc())
            recommendation_engine = None
    else:
        logger.warning("⚠ CSR Recommendation Engine module not available")
        logger.warning("   Check that csr_emotion_recommendations.py exists and has no errors")
    
    logger.info("="*70)
    logger.info("INITIALIZATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Transcriber:           {'✓ Ready' if transcriber else '✗ Not Available'}")
    logger.info(f"Feature Extractor:     {'✓ Ready' if feature_extractor else '✗ Not Available'}")
    logger.info(f"Emotion Classifier:    {'✓ Ready' if emotion_classifier else '✗ Not Available'}")
    logger.info(f"State Classifier:      {'✓ Ready' if state_classifier else '✗ Not Available'}")
    logger.info(f"Recommendation Engine: {'✓ Ready' if recommendation_engine else '✗ Not Available'}")
    logger.info("="*70)

# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models when API starts"""
    logger.info("Starting CSR Call Recording Analysis API...")
    try:
        initialize_models()
        logger.info("✓ API startup complete")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down CSR Call Recording Analysis API...")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_file(filepath: Path):
    """Clean up temporary files"""
    try:
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Cleaned up: {filepath}")
    except Exception as e:
        logger.warning(f"Could not cleanup {filepath}: {e}")

def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information and status"""
    return {
        "message": "CSR Call Recording Analysis API",
        "version": "1.0.0",
        "status": "running",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "modules_imported": modules_status,
        "models_initialized": {
            "transcriber": transcriber is not None,
            "feature_extractor": feature_extractor is not None,
            "emotion_classifier": emotion_classifier is not None,
            "state_classifier": state_classifier is not None,
            "recommendation_engine": recommendation_engine is not None
        },
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "transcribe": "/transcribe-only (POST)",
            "models_status": "/models/status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    core_models_ready = all([
        transcriber is not None,
        feature_extractor is not None,
        emotion_classifier is not None,
        state_classifier is not None
    ])
    
    return {
        "status": "healthy" if core_models_ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "transcriber": transcriber is not None,
            "feature_extractor": feature_extractor is not None,
            "emotion_classifier": emotion_classifier is not None,
            "state_classifier": state_classifier is not None,
            "recommendation_engine": recommendation_engine is not None
        },
        "ready_for_analysis": core_models_ready
    }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Main endpoint: Analyze audio file for emotion and generate CSR recommendations
    
    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
    
    Returns:
        Complete analysis with transcription, emotion detection, and CSR recommendations
    """
    
    # Validate required models are loaded
    required_models = {
        "transcriber": transcriber,
        "feature_extractor": feature_extractor,
        "emotion_classifier": emotion_classifier,
        "state_classifier": state_classifier
    }
    
    missing_models = [name for name, model in required_models.items() if model is None]
    
    if missing_models:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready. Missing models: {', '.join(missing_models)}"
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{file_ext}"
    
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Save uploaded file
        save_upload_file(file, temp_file)
        logger.info(f"File saved to: {temp_file}")
        
        # STEP 1: Transcribe audio
        logger.info("Step 1/5: Transcribing audio...")
        transcription_result = transcriber.transcribe_audio(str(temp_file))
        
        if not transcription_result or 'error' in transcription_result:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {transcription_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"✓ Transcription complete: {len(transcription_result.get('text', ''))} characters")
        
        # STEP 2: Extract MFCC features
        logger.info("Step 2/5: Extracting audio features...")
        features = feature_extractor.extract_features(str(temp_file))
        
        if features is None:
            raise HTTPException(
                status_code=500,
                detail="Feature extraction failed"
            )
        
        logger.info(f"✓ Features extracted: {features.shape}")
        
        # STEP 3: Predict emotion
        logger.info("Step 3/5: Classifying emotion...")
        prediction = emotion_classifier.predict(features)
        
        if not prediction or 'error' in prediction:
            raise HTTPException(
                status_code=500,
                detail=f"Emotion prediction failed: {prediction.get('error', 'Unknown error')}"
            )
        
        logger.info(f"✓ Emotion predicted: {prediction['emotion']} ({prediction['confidence']:.2%})")
        
        # STEP 4: Analyze emotional state
        logger.info("Step 4/5: Analyzing emotional state...")
        emotional_state = state_classifier.classify_state(prediction)
        logger.info(f"✓ Emotional state analyzed: {emotional_state['valence']}/{emotional_state['arousal']}")
        
        # STEP 5: Generate CSR recommendations
        logger.info("Step 5/5: Generating CSR recommendations...")
        recommendations = None
        
        if recommendation_engine:
            try:
                csr_emotional_state = recommendation_engine.classify_emotional_state(prediction)
                recommendations = recommendation_engine.generate_recommendation(csr_emotional_state)
                
                # Generate text report
                report = recommendation_engine.generate_report(csr_emotional_state, recommendations)
                recommendations['report'] = report
                
                # Save recommendation files
                saved_files = recommendation_engine.save_recommendation(
                    csr_emotional_state,
                    recommendations,
                    output_dir=str(OUTPUT_DIR / 'recommendations')
                )
                recommendations['saved_files'] = saved_files
                
                logger.info(f"✓ CSR recommendations generated and saved")
                
            except Exception as e:
                logger.error(f"Failed to generate recommendations: {e}")
                logger.error(traceback.format_exc())
                recommendations = {
                    "available": False,
                    "error": str(e),
                    "message": "Recommendation generation failed but analysis completed"
                }
        else:
            recommendations = {
                "available": False,
                "message": "Recommendation engine not initialized"
            }
        
        # Compile complete results
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "transcription": {
                "text": transcription_result.get('text', ''),
                "language": transcription_result.get('language', 'en'),
                "segments": transcription_result.get('segments', []),
                "duration": transcription_result.get('duration', 0),
                "processing_time": transcription_result.get('processing_time', 0)
            },
            "emotion_analysis": {
                "predicted_emotion": prediction['emotion'],
                "confidence": float(prediction['confidence']),
                "all_probabilities": prediction.get('all_probabilities', {}),
                "emotional_state": emotional_state
            },
            "csr_recommendations": recommendations,
            "metadata": {
                "audio_file": file.filename,
                "audio_duration": transcription_result.get('duration', 0),
                "total_processing_time": transcription_result.get('processing_time', 0)
            }
        }
        
        # Save complete analysis result
        result_file = OUTPUT_DIR / f"analysis_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✓ Complete analysis saved to: {result_file}")
        logger.info(f"✓ Analysis complete for {file.filename}")
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        cleanup_file(temp_file)
        raise
        
    except Exception as e:
        cleanup_file(temp_file)
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/transcribe-only")
async def transcribe_only(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Transcribe audio file without emotion analysis (faster)
    
    Args:
        file: Audio file
    
    Returns:
        Transcription result only
    """
    
    if not transcriber:
        raise HTTPException(
            status_code=503,
            detail="Transcriber not initialized"
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(file.filename).suffix
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{file_ext}"
    
    try:
        save_upload_file(file, temp_file)
        logger.info(f"Transcribing: {file.filename}")
        
        result = transcriber.transcribe_audio(str(temp_file))
        
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "transcription": result
        })
        
    except Exception as e:
        cleanup_file(temp_file)
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/models/status")
async def models_status():
    """Get detailed status of all models and modules"""
    return {
        "modules_imported": modules_status,
        "models_initialized": {
            "transcriber": {
                "loaded": transcriber is not None,
                "model_size": getattr(transcriber, 'model_size', None) if transcriber else None,
                "language": getattr(transcriber, 'language', None) if transcriber else None
            },
            "feature_extractor": {
                "loaded": feature_extractor is not None,
                "n_mfcc": getattr(feature_extractor, 'n_mfcc', None) if feature_extractor else None,
                "sample_rate": getattr(feature_extractor, 'sample_rate', None) if feature_extractor else None
            },
            "emotion_classifier": {
                "loaded": emotion_classifier is not None,
                "model_trained": getattr(emotion_classifier, 'model', None) is not None if emotion_classifier else False,
                "emotions": getattr(emotion_classifier, 'emotions', None) if emotion_classifier else None
            },
            "state_classifier": {
                "loaded": state_classifier is not None,
                "emotions_supported": len(getattr(state_classifier, 'EMOTION_CATEGORIES', {})) if state_classifier else 0
            },
            "recommendation_engine": {
                "loaded": recommendation_engine is not None,
                "emotion_profiles": len(getattr(recommendation_engine, 'EMOTION_PROFILES', {})) if recommendation_engine else 0,
                "communication_strategies": len(getattr(recommendation_engine, 'COMMUNICATION_STRATEGIES', {})) if recommendation_engine else 0
            }
        },
        "directories": {
            "uploads": str(UPLOAD_DIR),
            "outputs": str(OUTPUT_DIR),
            "models": str(MODELS_DIR)
        }
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CSR CALL RECORDING ANALYSIS API")
    print("="*70)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )