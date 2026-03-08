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

# ============================================================================
# SUPABASE STORAGE (soft import)
# ============================================================================

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

supabase_client = None
try:
    from supabase import create_client
    _supa_url = os.getenv("SUPABASE_URL")
    _supa_key = os.getenv("SUPABASE_SERVICE_KEY")
    if _supa_url and _supa_key and "YOUR-PROJECT" not in _supa_url:
        supabase_client = create_client(_supa_url, _supa_key)
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
# ML MODULES — imported lazily inside initialize_models() (background thread)
# so they don't block uvicorn from binding the port on startup
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
    """Load whisper + torch on first /analyze call (too heavy for startup on free tier)."""
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


def initialize_models():
    global feature_extractor, emotion_classifier, state_classifier, recommendation_engine

    # Whisper/torch skipped here — loaded lazily in /analyze to avoid OOM on 512MB.
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
        if supabase_client:
            try:
                import io, pickle
                print("⬇️  Loading svm_emotion_model.pkl from Supabase Storage…")
                res = supabase_client.storage.from_("models").download("svm_emotion_model.pkl")
                model_data = pickle.load(io.BytesIO(res))
                emotion_classifier.model = model_data["model"]
                emotion_classifier.scaler = model_data["scaler"]
                emotion_classifier.label_encoder = model_data["label_encoder"]
                emotion_classifier.classifier_type = model_data["classifier_type"]
                emotion_classifier.training_history = model_data["training_history"]
                emotion_classifier.is_trained = model_data["is_trained"]
                print("✅  Model loaded from Supabase Storage.")
            except Exception as dl_err:
                print(f"⚠️  Could not load model from Supabase: {dl_err}")
        else:
            print("⚠️  Supabase client not configured — emotion model not loaded.")
        print("✅  emotion_classifier loaded")
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
    import threading
    thread = threading.Thread(target=initialize_models, daemon=True)
    thread.start()


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
    ready = all([transcriber, feature_extractor, emotion_classifier, state_classifier])
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
    # Load whisper lazily (torch/whisper too large for startup on free tier)
    if not load_whisper():
        raise HTTPException(503, detail="Whisper/transcriber failed to load")

    required = {
        "feature_extractor": feature_extractor,
        "emotion_classifier": emotion_classifier,
        "state_classifier": state_classifier,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise HTTPException(503, detail=f"Service not ready. Missing: {', '.join(missing)}")

    if not file.filename:
        raise HTTPException(400, detail="No file provided")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{Path(file.filename).suffix}"

    try:
        save_upload_file(file, temp_file)

        transcription_result = transcriber.transcribe_call(str(temp_file))
        if not transcription_result or "error" in transcription_result:
            raise HTTPException(500, detail=f"Transcription failed: {transcription_result.get('error', 'Unknown')}")

        speaker_info = transcriber.detect_speakers(str(temp_file))
        features = feature_extractor.extract_features(speaker_info["caller_audio_path"])
        if features is None:
            raise HTTPException(500, detail="Feature extraction failed")

        prediction = emotion_classifier.predict_single(features)
        if not prediction or "error" in prediction:
            raise HTTPException(500, detail=f"Emotion prediction failed: {prediction.get('error', 'Unknown')}")

        emotional_state = state_classifier.classify_emotional_state(prediction)

        recommendations = {"available": False, "message": "Recommendation engine not initialized"}
        if recommendation_engine:
            try:
                csr_state = recommendation_engine.classify_emotional_state(prediction)
                recommendations = recommendation_engine.generate_recommendation(csr_state)
                recommendations["report"] = recommendation_engine.generate_report(csr_state, recommendations)
                recommendations["saved_files"] = recommendation_engine.save_recommendation(
                    csr_state, recommendations, output_dir=str(OUTPUT_DIR / "recommendations")
                )
            except Exception as e:
                recommendations = {"available": False, "error": str(e)}

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

        # Persist to DB if agent_id provided
        if db_available and agent_id:
            try:
                db: Session = SessionLocal()
                try:
                    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
                    if agent:
                        emotion_data = result["emotion_analysis"]
                        trans_data   = result["transcription"]
                        spk_data     = result["speaker_detection"]
                        rec_data     = result.get("csr_recommendations", {})

                        risk_map = {
                            "angry": "Critical", "frustrated": "High",
                            "sad": "Medium", "neutral": "Low",
                            "happy": "Low", "satisfied": "Low",
                        }
                        risk_level = risk_map.get(emotion_data["predicted_emotion"], "Low")

                        call_record = models.Call(
                            uuid=str(uuid_lib.uuid4()),
                            agent_id=agent.id,
                            cluster_id=agent.cluster_id,
                            filename=file.filename,
                            file_path=str(temp_file),
                            file_size=temp_file.stat().st_size if temp_file.exists() else None,
                            duration_sec=int(trans_data.get("duration", 0)) or None,
                            upload_status="analyzed",
                            call_date=date.today(),
                        )
                        db.add(call_record)
                        db.flush()

                        ar = models.AnalysisResult(
                            call_id=call_record.id,
                            predicted_emotion=emotion_data["predicted_emotion"],
                            confidence=emotion_data["confidence"],
                            all_probabilities=emotion_data.get("all_probabilities"),
                            valence=emotion_data.get("emotional_state", {}).get("valence"),
                            arousal=emotion_data.get("emotional_state", {}).get("arousal"),
                            risk_level=risk_level,
                            transcription_text=trans_data.get("text"),
                            transcription_lang=trans_data.get("language", "en"),
                            transcription_duration=trans_data.get("duration"),
                            speaker_mode=spk_data.get("mode"),
                            agent_channel=spk_data.get("agent_channel"),
                            caller_channel=spk_data.get("caller_channel"),
                        )
                        db.add(ar)
                        db.flush()

                        if rec_data.get("available") is not False:
                            action_req = rec_data.get("action_required", {})
                            comm       = rec_data.get("communication_guidance", {})
                            dd         = rec_data.get("dos_and_donts", {})
                            db.add(models.CSRRecommendation(
                                analysis_result_id=ar.id,
                                action=action_req.get("action", "NONE"),
                                urgency=action_req.get("urgency", "LOW"),
                                reason=action_req.get("reason"),
                                instruction=action_req.get("instruction"),
                                action_color=action_req.get("color"),
                                recommended_tone=comm.get("recommended_tone"),
                                example_phrases=comm.get("example_phrases"),
                                do_list=dd.get("do"),
                                dont_list=dd.get("dont"),
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

        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
            if speaker_info["mode"] == "stereo" and speaker_info["caller_audio_path"] != str(temp_file):
                background_tasks.add_task(cleanup_file, Path(speaker_info["caller_audio_path"]))

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
        result = transcriber.transcribe_call(str(temp_file))
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        return JSONResponse(content={"success": True, "filename": file.filename, "transcription": result})
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
    region: str
    overall_risk: Optional[str] = "Safe"

class ClusterUpdate(BaseModel):
    name: Optional[str] = None
    region: Optional[str] = None
    overall_risk: Optional[str] = None

class AgentCreate(BaseModel):
    cluster_id: int
    name: str
    email: str
    role: Optional[str] = "CSR"
    risk_level: Optional[str] = "Safe"

class AgentUpdate(BaseModel):
    cluster_id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
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

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ============================================================================
# USERS
# ============================================================================

def _hash_password(plain: str) -> str:
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto").hash(plain)

def _verify_password(plain: str, hashed: str) -> bool:
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto").verify(plain, hashed)


@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return [
        {
            "id": u.id,
            "uuid": str(u.uuid),
            "name": u.name,
            "email": u.email,
            "role": u.role,
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
    return {
        "id": u.id,
        "uuid": str(u.uuid),
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


@app.post("/users", status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == payload.email).first():
        raise HTTPException(400, detail="Email already in use")
    if payload.role not in ("agent", "supervisor"):
        raise HTTPException(400, detail="role must be 'agent' or 'supervisor'")
    u = models.User(
        name=payload.name,
        email=payload.email,
        password_hash=_hash_password(payload.password),
        role=payload.role,
    )
    db.add(u); db.commit(); db.refresh(u)
    return {
        "id": u.id,
        "uuid": str(u.uuid),
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
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
    if "role" in data and data["role"] not in ("agent", "supervisor"):
        raise HTTPException(400, detail="role must be 'agent' or 'supervisor'")
    for k, v in data.items():
        setattr(u, k, v)
    db.commit(); db.refresh(u)
    return {
        "id": u.id,
        "uuid": str(u.uuid),
        "name": u.name,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
        "updated_at": u.updated_at.isoformat() if u.updated_at else None,
    }


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.id == user_id).first()
    if not u:
        raise HTTPException(404, detail="User not found")
    db.delete(u); db.commit()


@app.post("/auth/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.email == email).first()
    if not u or not _verify_password(password, u.password_hash):
        raise HTTPException(401, detail="Invalid email or password")
    if not u.is_active:
        raise HTTPException(403, detail="Account is disabled")
    u.last_login_at = datetime.utcnow()
    db.commit()
    return {
        "id": u.id,
        "uuid": str(u.uuid),
        "name": u.name,
        "email": u.email,
        "role": u.role,
    }


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
        agents = db.query(models.Agent).filter(models.Agent.cluster_id == c.id).all()
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
    c = db.query(models.Cluster).filter(models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    return c


@app.post("/clusters", status_code=201)
def create_cluster(payload: ClusterCreate, db: Session = Depends(get_db)):
    if db.query(models.Cluster).filter(models.Cluster.name == payload.name).first():
        raise HTTPException(400, detail="Cluster name already exists")
    c = models.Cluster(**payload.model_dump())
    db.add(c); db.commit(); db.refresh(c)
    return c


@app.put("/clusters/{cluster_id}")
def update_cluster(cluster_id: int, payload: ClusterUpdate, db: Session = Depends(get_db)):
    c = db.query(models.Cluster).filter(models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    db.commit(); db.refresh(c)
    return c


@app.delete("/clusters/{cluster_id}", status_code=204)
def delete_cluster(cluster_id: int, db: Session = Depends(get_db)):
    c = db.query(models.Cluster).filter(models.Cluster.id == cluster_id).first()
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    db.delete(c); db.commit()


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
    if db.query(models.Agent).filter(models.Agent.email == payload.email).first():
        raise HTTPException(400, detail="Email already in use")
    if not db.query(models.Cluster).filter(models.Cluster.id == payload.cluster_id).first():
        raise HTTPException(404, detail="Cluster not found")
    a = models.Agent(**payload.model_dump())
    db.add(a); db.commit(); db.refresh(a)
    return a


@app.put("/agents/{agent_id}")
def update_agent(agent_id: int, payload: AgentUpdate, db: Session = Depends(get_db)):
    a = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not a:
        raise HTTPException(404, detail="Agent not found")
    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(a, k, v)
    db.commit(); db.refresh(a)
    return a


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    a = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not a:
        raise HTTPException(404, detail="Agent not found")
    db.delete(a); db.commit()


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
        joinedload(models.Call.analysis_result).joinedload(models.AnalysisResult.recommendation),
    )
    if cluster_id:
        q = q.filter(models.Call.cluster_id == cluster_id)
    if agent_id:
        q = q.filter(models.Call.agent_id == agent_id)

    result = []
    for c in q.order_by(models.Call.created_at.desc()).all():
        ar  = c.analysis_result
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
    db.commit(); db.refresh(c)
    return c


@app.delete("/calls/{call_id}", status_code=204)
def delete_call(call_id: int, db: Session = Depends(get_db)):
    c = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not c:
        raise HTTPException(404, detail="Call not found")
    db.delete(c); db.commit()


# ============================================================================
# REPORTS
# ============================================================================

@app.get("/reports/summary")
def reports_summary(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q_calls = db.query(func.count(models.Call.id))
    q_ar    = db.query(func.count(models.AnalysisResult.id))
    q_esc   = db.query(func.count(models.Escalation.id))
    if cluster_id:
        q_calls = q_calls.filter(models.Call.cluster_id == cluster_id)
        q_ar    = q_ar.join(models.Call).filter(models.Call.cluster_id == cluster_id)
        q_esc   = q_esc.join(models.Call, models.Escalation.call_id == models.Call.id).filter(models.Call.cluster_id == cluster_id)
    return {
        "total_calls": q_calls.scalar() or 0,
        "analyzed_calls": q_ar.scalar() or 0,
        "escalations": q_esc.scalar() or 0,
        "total_agents": db.query(func.count(models.Agent.id)).scalar() or 0,
        "risky_agents": db.query(func.count(models.Agent.id)).filter(models.Agent.risk_level == "Risky").scalar() or 0,
    }


@app.get("/reports/emotion-distribution")
def reports_emotion_distribution(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func
    q = db.query(models.AnalysisResult.predicted_emotion, func.count(models.AnalysisResult.id).label("count"))
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
        extract("year",  models.Call.call_date).label("year"),
        extract("month", models.Call.call_date).label("month"),
        models.AnalysisResult.predicted_emotion,
        func.count(models.AnalysisResult.id).label("count"),
    ).join(models.AnalysisResult, models.AnalysisResult.call_id == models.Call.id).filter(models.Call.call_date >= cutoff)
    if cluster_id:
        q = q.filter(models.Call.cluster_id == cluster_id)
    rows = q.group_by("year", "month", models.AnalysisResult.predicted_emotion).all()
    pivot: dict = {}
    for r in rows:
        label = f"{int(r.year)}-{int(r.month):02d}"
        pivot.setdefault(label, {"month": label})[r.predicted_emotion] = r.count
    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/risk-trend")
def reports_risk_trend(cluster_id: Optional[int] = None, db: Session = Depends(get_db)):
    from sqlalchemy import func, extract
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=365)
    q = db.query(
        extract("year",  models.Call.call_date).label("year"),
        extract("month", models.Call.call_date).label("month"),
        models.AnalysisResult.risk_level,
        func.count(models.AnalysisResult.id).label("count"),
    ).join(models.AnalysisResult, models.AnalysisResult.call_id == models.Call.id).filter(models.Call.call_date >= cutoff)
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
        extract("year",  models.Call.call_date).label("year"),
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
            models.AnalysisResult.risk_level, func.count(models.AnalysisResult.id).label("cnt")
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
                (rm.get("Critical", 0)*4 + rm.get("High", 0)*3 + rm.get("Medium", 0)*2 + rm.get("Low", 0))
                / total * 25, 1
            ),
        })
    return sorted(result, key=lambda x: -x["risk_score"])


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
