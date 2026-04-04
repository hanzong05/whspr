"""
CSR Call Recording Analysis API
FastAPI backend — ML pipeline + CRUD endpoints
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
import uvicorn
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime, date
import json
import uuid as uuid_lib
from sqlalchemy.orm import Session
from pydub import AudioSegment
import bcrypt
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

# ============================================================================
# PYTHON PATH
# ============================================================================

CURRENT_DIR = Path(__file__).parent.absolute()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ============================================================================
# DATABASE (soft import — API still works without DB)
# ============================================================================

db_available = False
try:
    from database import get_db, SessionLocal
    import models
    db_available = True
except Exception as _db_err:
    print(f"⚠️  Database not available: {_db_err}")

# ============================================================================
# ML MODULES
# ============================================================================

modules_status = {
    "whisper": False,
    "mfcc": False,
    "emotion_classifier": False,
    "state_classifier": False,
    "recommendation_engine": False,
}

# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="CSR Call Recording Analysis API",
    description="Emotion detection and CSR recommendation system for call recordings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DIRECTORIES
# ============================================================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

for _d in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, OUTPUT_DIR / "recommendations"]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ML MODEL INSTANCES
# ============================================================================

transcriber = None
feature_extractor = None
emotion_classifier = None
state_classifier = None
recommendation_engine = None


def load_whisper():
    global transcriber
    if transcriber is not None:
        return True
    try:
        from whisper_asr_module import CSRCallTranscriber
        transcriber = CSRCallTranscriber(model_size="base", language="en")
        modules_status["whisper"] = True
        print("✅  whisper loaded (lazy)")
        return True
    except Exception as e:
        print(f"❌  whisper failed: {e}")
        return False


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any audio file to WAV using multiple strategies.
    Returns the WAV path on success, or the original path as fallback.
    """
    if input_path.suffix.lower() == ".wav":
        return input_path

    wav_path = input_path.with_suffix(".wav")

    # Strategy 1: ogg (Chrome/Edge send opus-in-ogg or opus-in-webm)
    try:
        audio = AudioSegment.from_file(str(input_path), format="ogg")
        audio.export(str(wav_path), format="wav")
        print("✅ WAV OK (ogg)")
        return wav_path
    except Exception as e1:
        print(f"⚠️ ogg failed: {e1}")

    # Strategy 2: explicit webm
    try:
        audio = AudioSegment.from_file(str(input_path), format="webm")
        audio.export(str(wav_path), format="wav")
        print("✅ WAV OK (webm)")
        return wav_path
    except Exception as e2:
        print(f"⚠️ webm failed: {e2}")

    # Strategy 3: auto-detect
    try:
        audio = AudioSegment.from_file(str(input_path))
        audio.export(str(wav_path), format="wav")
        print("✅ WAV OK (auto)")
        return wav_path
    except Exception as e3:
        print(f"⚠️ All conversions failed, using original: {e3}")
        return input_path


def ensure_wav(input_path: Path) -> Path:
    """Legacy helper — delegates to convert_to_wav."""
    return convert_to_wav(input_path)


def initialize_models():
    global feature_extractor, emotion_classifier, state_classifier, recommendation_engine

    try:
        from mfcc_feature_extraction import MFCCFeatureExtractor as _MFCC
        feature_extractor = _MFCC()
        modules_status["mfcc"] = True
        print("✅  mfcc loaded")
    except Exception as e:
        print(f"❌  mfcc failed: {e}")

    try:
        from ml_classifier import EmotionClassifier as _EC
        emotion_classifier = _EC()
        modules_status["emotion_classifier"] = True
        model_path = MODELS_DIR / "svm_emotion_model.pkl"
        if model_path.exists():
            emotion_classifier.load_model(str(model_path))
            print("✅  emotion_classifier loaded")
        else:
            print(f"⚠️  Model file not found at {model_path}")
    except Exception as e:
        print(f"❌  emotion_classifier failed: {e}")

    try:
        from emotional_state_classifier import EmotionalStateClassifier as _ESC
        state_classifier = _ESC()
        modules_status["state_classifier"] = True
        print("✅  state_classifier loaded")
    except Exception as e:
        print(f"❌  state_classifier failed: {e}")

    try:
        from csr_emotion_recommendations import CSREmotionClassifier as _CEC
        recommendation_engine = _CEC()
        modules_status["recommendation_engine"] = True
        print("✅  recommendation_engine loaded")
    except Exception as e:
        print(f"❌  recommendation_engine failed: {e}")


@app.on_event("startup")
async def startup_event():
    global transcriber

    print("🚀 Loading ALL models at startup...")

    # ✅ Load Whisper (REMOVE lazy)
    try:
        from whisper_asr_module import CSRCallTranscriber
        transcriber = CSRCallTranscriber(model_size="base", language="en")
        modules_status["whisper"] = True
        print("✅ Whisper loaded")
    except Exception as e:
        print(f"❌ Whisper failed: {e}")

    # ✅ Load other models
    initialize_models()

    print("🔥 ALL MODELS READY")


# ============================================================================
# UTILITIES
# ============================================================================

def cleanup_file(filepath: Path):
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception:
        pass


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()


# ============================================================================
# PASSWORD HELPERS
# ============================================================================

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ============================================================================
# ML ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "CSR Call Recording Analysis API",
        "version": "1.0.0",
        "status": "running",
        "db_connected": db_available,
        "modules": modules_status,
    }


@app.get("/health")
async def health_check():
    ready = all([transcriber, feature_extractor,
                emotion_classifier, state_classifier])
    return {
        "status": "healthy" if ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "transcriber": transcriber is not None,
            "feature_extractor": feature_extractor is not None,
            "emotion_classifier": emotion_classifier is not None,
            "state_classifier": state_classifier is not None,
            "recommendation_engine": recommendation_engine is not None,
        },
        "ready_for_analysis": ready,
    }


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    agent_id: Optional[int] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    if not transcriber:
        raise HTTPException(503, detail="Whisper not loaded")

    required = {
        "feature_extractor": feature_extractor,
        "emotion_classifier": emotion_classifier,
        "state_classifier": state_classifier,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise HTTPException(
            503, detail=f"Service not ready. Missing: {', '.join(missing)}")

    if not file.filename:
        raise HTTPException(400, detail="No file provided")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{Path(file.filename).suffix}"

    try:
        save_upload_file(file, temp_file)

        # ── Step 1: Convert uploaded file to WAV ──────────────────────────────
        process_path = convert_to_wav(temp_file)

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        transcription_result = transcriber.transcribe_call(str(process_path))

        if not transcription_result or "error" in transcription_result:
            raise HTTPException(
                500, detail=f"Transcription failed: {transcription_result.get('error', 'Unknown')}"
            )

        # ── Step 3: Speaker diarization (uses the already-converted WAV) ──────
        speaker_info = transcriber.detect_speakers(str(process_path))

        caller_audio_path = Path(speaker_info["caller_audio_path"])

        if not caller_audio_path.exists() or caller_audio_path.stat().st_size == 0:
            raise HTTPException(
                500, detail=f"Caller audio split missing or empty: {caller_audio_path}"
            )

        # Only re-convert if the split file is not already WAV
        if caller_audio_path.suffix.lower() != ".wav":
            caller_audio_wav = convert_to_wav(caller_audio_path)
        else:
            caller_audio_wav = caller_audio_path

        if not caller_audio_wav.exists() or caller_audio_wav.stat().st_size == 0:
            raise HTTPException(
                500, detail=f"Caller WAV empty after conversion: {caller_audio_wav}"
            )

        speaker_info["caller_audio_path"] = str(caller_audio_wav)

        # ── Step 4: Feature extraction ────────────────────────────────────────
        features = feature_extractor.extract_features(str(caller_audio_wav))

        if features is None:
            raise HTTPException(
                500, detail="Feature extraction returned None — caller audio may be too short or silent"
            )

        # ── Step 5: Emotion prediction ────────────────────────────────────────
        prediction = emotion_classifier.predict_single(features)

        if not prediction or "error" in prediction:
            raise HTTPException(
                500, detail=f"Emotion prediction failed: {prediction.get('error', 'Unknown')}"
            )

        # ── Step 6: Emotional state classification ────────────────────────────
        emotional_state = state_classifier.classify_emotional_state(prediction)

        # ── Step 7: Recommendations ───────────────────────────────────────────
        recommendations = {"available": False,
                           "message": "Recommendation engine not initialized"}
        if recommendation_engine:
            try:
                csr_state = recommendation_engine.classify_emotional_state(
                    prediction)
                recommendations = recommendation_engine.generate_recommendation(
                    csr_state)
                recommendations["report"] = recommendation_engine.generate_report(
                    csr_state, recommendations)
                recommendations["saved_files"] = recommendation_engine.save_recommendation(
                    csr_state, recommendations, output_dir=str(
                        OUTPUT_DIR / "recommendations")
                )
            except Exception as e:
                recommendations = {"available": False, "error": str(e)}

        # ── Step 8: Build result ──────────────────────────────────────────────
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "transcription": {
                "text": transcription_result.get("text", ""),
                "language": transcription_result.get("language", "en"),
                "segments": transcription_result.get("segments", []),
                "duration": transcription_result.get("duration", 0),
            },
            "speaker_detection": {
                "mode": speaker_info["mode"],
                "agent_channel": speaker_info.get("agent_channel"),
                "caller_channel": speaker_info.get("caller_channel"),
                "message": speaker_info["message"],
            },
            "emotion_analysis": {
                "predicted_emotion": prediction["emotion"],
                "confidence": float(prediction["confidence"]),
                "all_probabilities": prediction.get("all_probabilities", {}),
                "emotional_state": emotional_state,
            },
            "csr_recommendations": recommendations,
        }

        with open(OUTPUT_DIR / f"analysis_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # ── Step 9: Persist to DB ─────────────────────────────────────────────
        if db_available and agent_id:
            try:
                db: Session = SessionLocal()
                try:
                    agent = db.query(models.Agent).filter(
                        models.Agent.id == agent_id).first()
                    if agent:
                        emotion_data = result["emotion_analysis"]
                        trans_data = result["transcription"]
                        spk_data = result["speaker_detection"]
                        rec_data = result.get("csr_recommendations", {})

                        risk_map = {
                            "angry": "Critical", "frustrated": "High",
                            "sad": "Medium", "neutral": "Low",
                            "happy": "Low", "satisfied": "Low",
                        }
                        risk_level = risk_map.get(
                            emotion_data["predicted_emotion"], "Low")

                        call_record = models.Call(
                            uuid=str(uuid_lib.uuid4()),
                            agent_id=agent.id,
                            cluster_id=agent.cluster_id,
                            filename=file.filename,
                            file_path=str(temp_file),
                            file_size=temp_file.stat().st_size if temp_file.exists() else None,
                            duration_sec=int(trans_data.get(
                                "duration", 0)) or None,
                            upload_status="analyzed",
                            call_date=date.today(),
                        )
                        db.add(call_record)
                        db.flush()

                        ar = models.AnalysisResult(
                            call_id=call_record.id,
                            predicted_emotion=emotion_data["predicted_emotion"],
                            confidence=emotion_data["confidence"],
                            all_probabilities=json.dumps(emotion_data.get(
                                "all_probabilities") or {}),  # ← fixed
                            valence=emotion_data.get(
                                "emotional_state", {}).get("valence"),
                            arousal=emotion_data.get(
                                "emotional_state", {}).get("arousal"),
                            risk_level=risk_level,
                            transcription_text=trans_data.get("text"),
                            transcription_lang=trans_data.get(
                                "language", "en"),
                            transcription_duration=trans_data.get("duration"),
                            speaker_mode=spk_data.get("mode"),
                            agent_channel=spk_data.get("agent_channel"),
                            caller_channel=spk_data.get("caller_channel"),
                        )

                        db.add(ar)
                        db.flush()

                        if rec_data.get("available") is not False:
                            action_req = rec_data.get("action_required", {})
                            comm = rec_data.get("communication_guidance", {})
                            dd = rec_data.get("dos_and_donts", {})
                            db.add(models.CSRRecommendation(
                                analysis_result_id=ar.id,
                                action=action_req.get("action", "NONE"),
                                urgency=action_req.get("urgency", "LOW"),
                                reason=action_req.get("reason"),
                                instruction=action_req.get("instruction"),
                                action_color=action_req.get("color"),
                                recommended_tone=comm.get("recommended_tone"),
                                example_phrases=json.dumps(
                                    comm.get("example_phrases") or []),
                                do_list=json.dumps(dd.get("do") or []),
                                dont_list=json.dumps(dd.get("dont") or []),
                            ))

                        agent.risk_level = (
                            "Risky" if risk_level in ("Critical", "High")
                            else "Medium" if risk_level == "Medium"
                            else agent.risk_level
                        )
                        db.commit()
                        result["call_id"] = call_record.id
                finally:
                    db.close()
            except Exception as db_err:
                print(f"⚠️  DB persist error: {db_err}")

        # ── Step 10: Cleanup temp files ───────────────────────────────────────
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
            if speaker_info["mode"] == "stereo" and speaker_info["caller_audio_path"] != str(temp_file):
                background_tasks.add_task(cleanup_file, Path(
                    speaker_info["caller_audio_path"]))

        return JSONResponse(content=result)

    except HTTPException:
        cleanup_file(temp_file)
        raise
    except Exception as e:
        cleanup_file(temp_file)
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")


@app.post("/transcribe-only")
async def transcribe_only(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not transcriber:
        raise HTTPException(503, detail="Transcriber not initialized")
    if not file.filename:
        raise HTTPException(400, detail="No file provided")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{Path(file.filename).suffix}"

    try:
        save_upload_file(file, temp_file)
        process_path = convert_to_wav(temp_file)
        transcription_result = transcriber.transcribe_call(str(process_path))

        if not transcription_result or "error" in transcription_result:
            raise HTTPException(
                500, detail=f"Transcription failed: {transcription_result.get('error', 'Unknown')}"
            )

        speaker_info = transcriber.detect_speakers(str(process_path))

        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "transcription": transcription_result,
            "speaker_detection": speaker_info,
        })
    except HTTPException:
        cleanup_file(temp_file)
        raise
    except Exception as e:
        cleanup_file(temp_file)
        raise HTTPException(500, detail=f"Transcription failed: {str(e)}")


@app.get("/models/status")
async def models_status_endpoint():
    return {
        "modules_imported": modules_status,
        "models_initialized": {
            "transcriber": transcriber is not None,
            "feature_extractor": feature_extractor is not None,
            "emotion_classifier": emotion_classifier is not None,
            "state_classifier": state_classifier is not None,
            "recommendation_engine": recommendation_engine is not None,
        },
    }


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ClusterCreate(BaseModel):
    name: str


class ClusterUpdate(BaseModel):
    name: Optional[str] = None


class AgentCreate(BaseModel):
    cluster_id: int
    name: str
    risk_level: Optional[str] = "Safe"


class AgentUpdate(BaseModel):
    cluster_id: Optional[int] = None
    name: Optional[str] = None
    risk_level: Optional[str] = None
    is_active: Optional[bool] = None


class CallUpdate(BaseModel):
    agent_id: Optional[int] = None
    call_date: Optional[date] = None
    upload_status: Optional[str] = None


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: Optional[str] = "agent"


class RoleOption(BaseModel):
    value: str
    label: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class RegisterPayload(BaseModel):
    agent_id: int
    password: str
    role: str


class LoginPayload(BaseModel):
    username: str
    password: str


# ============================================================================
# AUTH
# ============================================================================

@app.get("/roles")
def list_roles():
    return [
        {"value": "admin", "label": "Admin"},
        {"value": "supervisor", "label": "Supervisor"},
        {"value": "agent", "label": "Agent"},
    ]


@app.post("/auth/register", status_code=201)
def register(payload: RegisterPayload, db: Session = Depends(get_db)):
    agent = db.query(models.Agent).filter(
        models.Agent.id == payload.agent_id).first()
    if not agent:
        raise HTTPException(404, detail="Agent not found")

    existing = db.query(models.User).filter(
        models.User.agent_id == payload.agent_id).first()
    if existing:
        raise HTTPException(
            400, detail="An account already exists for this agent")

    if len(payload.password) < 8:
        raise HTTPException(
            400, detail="Password must be at least 8 characters")

    allowed_roles = {"admin", "supervisor", "agent"}
    if payload.role not in allowed_roles:
        raise HTTPException(400, detail="Invalid role selected")

    user = models.User(
        agent_id=payload.agent_id,
        username=str(payload.agent_id).zfill(4),
        password_hash=_hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "agent_id": user.agent_id,
        "username": user.username,
        "role": user.role,
        "agent_name": agent.name,
        "agent_email": agent.email,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@app.post("/auth/login")
def login(payload: LoginPayload, db: Session = Depends(get_db)):
    if not db_available:
        raise HTTPException(503, detail="Database not available")

    user = db.query(models.User).filter(
        models.User.username == payload.username).first()

    if not user:
        agent = db.query(models.Agent).filter(
            models.Agent.email == payload.username).first()
        if agent:
            user = db.query(models.User).filter(
                models.User.agent_id == agent.id).first()

    if not user or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(401, detail="Invalid username or password")

    if not user.is_active:
        raise HTTPException(403, detail="Account is disabled")

    user.last_login_at = datetime.utcnow()
    db.commit()

    agent = db.query(models.Agent).filter(
        models.Agent.id == user.agent_id).first()

    return {
        "id": user.id,
        "agent_id": user.agent_id,
        "username": user.username,
        "agent_name": agent.name if agent else None,
        "agent_email": agent.email if agent else None,
        "role": user.role,                          # ← this is what frontend needs
        "cluster_id": agent.cluster_id if agent else None,
        "cluster_name": agent.cluster.name if (agent and agent.cluster) else None,
        "is_active": user.is_active,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


# ============================================================================
# USERS
# ============================================================================

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return [
        {
            "id": u.id,
            "agent_id": u.agent_id,
            "username": u.username,
            "is_active": u.is_active,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]


@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(404, detail="User not found")
    agent = db.query(models.Agent).filter(
        models.Agent.id == u.agent_id).first()
    return {
        "id": u.id,
        "agent_id": u.agent_id,
        "username": u.username,
        "is_active": u.is_active,
        "agent_name": agent.name if agent else None,
        "agent_email": agent.email if agent else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


@app.put("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(404, detail="User not found")
    data = payload.model_dump(exclude_none=True)
    if "password" in data:
        u.password_hash = _hash_password(data.pop("password"))
    for k, v in data.items():
        setattr(u, k, v)
    db.commit()
    db.refresh(u)
    return {
        "id": u.id,
        "agent_id": u.agent_id,
        "username": u.username,
        "is_active": u.is_active,
    }


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(404, detail="User not found")
    db.delete(u)
    db.commit()


# ============================================================================
# CLUSTERS
# ============================================================================

@app.get("/clusters")
def list_clusters(db: Session = Depends(get_db)):
    from sqlalchemy import func
    clusters = db.query(models.Cluster).all()
    result = []
    today = date.today()
    for c in clusters:
        agents = db.query(models.Agent).filter(
            models.Agent.cluster_id == c.id).all()
        calls_today = db.query(func.count(models.Call.id)).filter(
            models.Call.cluster_id == c.id, models.Call.call_date == today
        ).scalar() or 0
        total_calls = db.query(func.count(models.Call.id)).filter(
            models.Call.cluster_id == c.id
        ).scalar() or 0
        result.append({
            "id": c.id,
            "name": c.name,
            "region": c.region,
            "overall_risk": c.overall_risk,
            "agent_count": len(agents),
            "risky_agents": sum(1 for a in agents if a.risk_level == "Risky"),
            "medium_agents": sum(1 for a in agents if a.risk_level == "Medium"),
            "safe_agents": sum(1 for a in agents if a.risk_level == "Safe"),
            "calls_today": calls_today,
            "total_calls": total_calls,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })
    return result


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int, db: Session = Depends(get_db)):
    c = db.query(models.Cluster).filter(
        models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    return c


@app.post("/clusters", status_code=201)
def create_cluster(payload: ClusterCreate, db: Session = Depends(get_db)):
    if db.query(models.Cluster).filter(models.Cluster.name == payload.name).first():
        raise HTTPException(400, detail="Cluster name already exists")
    c = models.Cluster(**payload.model_dump())
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@app.put("/clusters/{cluster_id}")
def update_cluster(cluster_id: int, payload: ClusterUpdate, db: Session = Depends(get_db)):
    c = db.query(models.Cluster).filter(
        models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    return c


@app.delete("/clusters/{cluster_id}", status_code=204)
def delete_cluster(cluster_id: int, db: Session = Depends(get_db)):
    c = db.query(models.Cluster).filter(
        models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    db.delete(c)
    db.commit()


# ============================================================================
# AGENTS
# ============================================================================

@app.get("/agents")
def list_agents(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q = db.query(models.Agent)
    if cluster_id:
        q = q.filter(models.Agent.cluster_id == cluster_id)
    today = date.today()
    result = []
    for a in q.all():
        calls_today = db.query(func.count(models.Call.id)).filter(
            models.Call.agent_id == a.id, models.Call.call_date == today
        ).scalar() or 0
        total_calls = db.query(func.count(models.Call.id)).filter(
            models.Call.agent_id == a.id
        ).scalar() or 0
        result.append({
            "id": a.id,
            "name": a.name,
            "email": a.email,
            "role": a.role,
            "risk_level": a.risk_level,
            "is_active": a.is_active,
            "cluster_id": a.cluster_id,
            "cluster_name": a.cluster.name if a.cluster else None,
            "calls_today": calls_today,
            "total_calls": total_calls,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        })
    return result


@app.get("/agents/{agent_id}")
def get_agent(agent_id: int, db: Session = Depends(get_db)):
    a = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not a:
        raise HTTPException(404, detail="Agent not found")
    return a


@app.post("/agents", status_code=201)
def create_agent(payload: AgentCreate, db: Session = Depends(get_db)):
    if not db.query(models.Cluster).filter(models.Cluster.id == payload.cluster_id).first():
        raise HTTPException(404, detail="Cluster not found")

    a = models.Agent(**payload.model_dump())
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@app.put("/agents/{agent_id}")
def update_agent(agent_id: int, payload: AgentUpdate, db: Session = Depends(get_db)):
    a = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not a:
        raise HTTPException(404, detail="Agent not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(a, k, v)
    db.commit()
    db.refresh(a)
    return a


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    a = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not a:
        raise HTTPException(404, detail="Agent not found")
    db.delete(a)
    db.commit()


# ============================================================================
# CALLS
# ============================================================================

@app.get("/calls")
def list_calls(
    cluster_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    from sqlalchemy.orm import joinedload
    q = db.query(models.Call).options(
        joinedload(models.Call.agent),
        joinedload(models.Call.cluster),
        joinedload(models.Call.analysis_result).joinedload(
            models.AnalysisResult.recommendation),
    )
    if cluster_id:
        q = q.filter(models.Call.cluster_id == cluster_id)
    if agent_id:
        q = q.filter(models.Call.agent_id == agent_id)

    result = []
    for c in q.order_by(models.Call.created_at.desc()).all():
        ar = c.analysis_result
        rec = ar.recommendation if ar else None
        result.append({
            "id": c.id,
            "uuid": c.uuid,
            "filename": c.filename,
            "file_size": c.file_size,
            "duration_sec": c.duration_sec,
            "upload_status": c.upload_status,
            "call_date": c.call_date.isoformat() if c.call_date else None,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "agent": {"id": c.agent.id, "name": c.agent.name, "email": c.agent.email} if c.agent else None,
            "cluster": {"id": c.cluster.id, "name": c.cluster.name} if c.cluster else None,
            "analysis": {
                "predicted_emotion": ar.predicted_emotion,
                "confidence": float(ar.confidence),
                "risk_level": ar.risk_level,
                "transcription_text": ar.transcription_text,
                "valence": ar.valence,
                "arousal": ar.arousal,
            } if ar else None,
            "recommendation": {
                "action": rec.action,
                "urgency": rec.urgency,
                "reason": rec.reason,
                "instruction": rec.instruction,
                "action_color": rec.action_color,
                "recommended_tone": rec.recommended_tone,
                "example_phrases": rec.example_phrases,
                "do_list": rec.do_list,
                "dont_list": rec.dont_list,
            } if rec else None,
        })
    return result


@app.put("/calls/{call_id}")
def update_call(call_id: int, payload: CallUpdate, db: Session = Depends(get_db)):
    c = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not c:
        raise HTTPException(404, detail="Call not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    return c


@app.delete("/calls/{call_id}", status_code=204)
def delete_call(call_id: int, db: Session = Depends(get_db)):
    c = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not c:
        raise HTTPException(404, detail="Call not found")
    db.delete(c)
    db.commit()


# ============================================================================
# REPORTS
# ============================================================================

@app.get("/reports/summary")
def reports_summary(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q_calls = db.query(func.count(models.Call.id))
    q_ar = db.query(func.count(models.AnalysisResult.id))
    q_esc = db.query(func.count(models.Escalation.id))
    if cluster_id:
        q_calls = q_calls.filter(models.Call.cluster_id == cluster_id)
        q_ar = q_ar.join(models.Call).filter(
            models.Call.cluster_id == cluster_id)
        q_esc = q_esc.join(models.Call, models.Escalation.call_id == models.Call.id).filter(
            models.Call.cluster_id == cluster_id)
    return {
        "total_calls": q_calls.scalar() or 0,
        "analyzed_calls": q_ar.scalar() or 0,
        "escalations": q_esc.scalar() or 0,
        "total_agents": db.query(func.count(models.Agent.id)).scalar() or 0,
        "risky_agents": db.query(func.count(models.Agent.id)).filter(
            models.Agent.risk_level == "Risky").scalar() or 0,
    }


@app.get("/reports/emotion-distribution")
def reports_emotion_distribution(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q = db.query(
        models.AnalysisResult.predicted_emotion,
        func.count(models.AnalysisResult.id).label("count")
    )
    if cluster_id:
        q = q.join(models.Call).filter(models.Call.cluster_id == cluster_id)
    rows = q.group_by(models.AnalysisResult.predicted_emotion).all()
    return [{"emotion": r.predicted_emotion, "count": r.count} for r in rows]


@app.get("/reports/emotion-trend")
def reports_emotion_trend(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func, extract
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=365)
    q = db.query(
        extract("year", models.Call.call_date).label("year"),
        extract("month", models.Call.call_date).label("month"),
        models.AnalysisResult.predicted_emotion,
        func.count(models.AnalysisResult.id).label("count"),
    ).join(models.AnalysisResult, models.AnalysisResult.call_id == models.Call.id).filter(
        models.Call.call_date >= cutoff
    )
    if cluster_id:
        q = q.filter(models.Call.cluster_id == cluster_id)
    rows = q.group_by(
        "year", "month", models.AnalysisResult.predicted_emotion).all()
    pivot: dict = {}
    for r in rows:
        label = f"{int(r.year)}-{int(r.month):02d}"
        pivot.setdefault(label, {"month": label})[
            r.predicted_emotion] = r.count
    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/risk-trend")
def reports_risk_trend(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func, extract
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=365)
    q = db.query(
        extract("year", models.Call.call_date).label("year"),
        extract("month", models.Call.call_date).label("month"),
        models.AnalysisResult.risk_level,
        func.count(models.AnalysisResult.id).label("count"),
    ).join(models.AnalysisResult, models.AnalysisResult.call_id == models.Call.id).filter(
        models.Call.call_date >= cutoff
    )
    if cluster_id:
        q = q.filter(models.Call.cluster_id == cluster_id)
    rows = q.group_by("year", "month", models.AnalysisResult.risk_level).all()
    pivot: dict = {}
    for r in rows:
        label = f"{int(r.year)}-{int(r.month):02d}"
        pivot.setdefault(label, {"month": label})[r.risk_level] = r.count
    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/call-volume")
def reports_call_volume(db: Session = Depends(get_db)):
    from sqlalchemy import func, extract
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=365)
    rows = db.query(
        extract("year", models.Call.call_date).label("year"),
        extract("month", models.Call.call_date).label("month"),
        models.Cluster.name.label("cluster"),
        func.count(models.Call.id).label("count"),
    ).join(models.Cluster, models.Call.cluster_id == models.Cluster.id).filter(
        models.Call.call_date >= cutoff
    ).group_by("year", "month", models.Cluster.name).all()
    pivot: dict = {}
    for r in rows:
        label = f"{int(r.year)}-{int(r.month):02d}"
        pivot.setdefault(label, {"month": label})[r.cluster] = r.count
    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/agent-risk-scores")
def reports_agent_risk_scores(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q = db.query(models.Agent)
    if cluster_id:
        q = q.filter(models.Agent.cluster_id == cluster_id)
    result = []
    for a in q.all():
        counts = db.query(
            models.AnalysisResult.risk_level,
            func.count(models.AnalysisResult.id).label("cnt")
        ).join(models.Call, models.AnalysisResult.call_id == models.Call.id).filter(
            models.Call.agent_id == a.id
        ).group_by(models.AnalysisResult.risk_level).all()
        rm = {r.risk_level: r.cnt for r in counts}
        total = sum(rm.values()) or 1
        result.append({
            "agent_id": a.id,
            "agent_name": a.name,
            "cluster": a.cluster.name if a.cluster else None,
            "risk_level": a.risk_level,
            "critical": rm.get("Critical", 0),
            "high": rm.get("High", 0),
            "medium": rm.get("Medium", 0),
            "low": rm.get("Low", 0),
            "total_calls": total,
            "risk_score": round(
                (rm.get("Critical", 0) * 4 + rm.get("High", 0) * 3 +
                 rm.get("Medium", 0) * 2 + rm.get("Low", 0)) / total * 25, 1
            ),
        })
    return sorted(result, key=lambda x: -x["risk_score"])


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=True, log_level="info")
