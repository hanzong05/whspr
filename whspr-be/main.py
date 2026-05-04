"""
CSR Call Recording Analysis API
FastAPI backend — ML pipeline + CRUD endpoints (Supabase client)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
import shutil
import mimetypes
from pathlib import Path
from datetime import datetime, date, timedelta
import json
import uuid as uuid_lib
from pydub import AudioSegment
import bcrypt

try:
    from postgrest.exceptions import APIError as _PostgRESTAPIError
except Exception:
    _PostgRESTAPIError = Exception

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

# ── Supabase client ──────────────────────────────────────────────────────────
supabase_client = None
try:
    # Disable HTTP/2 globally to prevent WinError 10035 on Windows
    import httpx
    _orig_init = httpx.Client.__init__
    def _http1_init(self, *args, **kwargs):
        kwargs["http2"] = False
        _orig_init(self, *args, **kwargs)
    httpx.Client.__init__ = _http1_init

    from supabase import create_client
    _supa_url = os.getenv("SUPABASE_URL")
    _supa_key = os.getenv("SUPABASE_SERVICE_KEY")
    if _supa_url and _supa_key:
        supabase_client = create_client(_supa_url, _supa_key)
        print("✅ Supabase client ready (HTTP/1.1)")
    else:
        print("⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
except Exception as _e:
    print(f"⚠️  Supabase client init failed: {_e}")

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audio-recordings")
db_available = supabase_client is not None


# ── DB helpers ───────────────────────────────────────────────────────────────

def _require_db():
    if not supabase_client:
        raise HTTPException(503, detail="Database not available")


def _get_all(table: str, select: str = "*") -> list:
    return supabase_client.table(table).select(select).execute().data or []


def _get_one(table: str, id_val, id_col: str = "id", select: str = "*"):
    rows = supabase_client.table(table).select(select).eq(id_col, id_val).execute().data
    return rows[0] if rows else None


def _get_by(table: str, filters: dict, select: str = "*") -> list:
    q = supabase_client.table(table).select(select)
    for col, val in filters.items():
        q = q.eq(col, val)
    return q.execute().data or []


def _count(table: str, filters: Optional[dict] = None) -> int:
    q = supabase_client.table(table).select("id", count="exact")
    if filters:
        for col, val in filters.items():
            q = q.eq(col, val)
    return q.execute().count or 0


def _count_date(table: str, date_col: str, date_val: str, extra: Optional[dict] = None) -> int:
    q = supabase_client.table(table).select("id", count="exact").eq(date_col, date_val)
    if extra:
        for col, val in extra.items():
            q = q.eq(col, val)
    return q.execute().count or 0


def _insert(table: str, data: dict) -> dict:
    result = supabase_client.table(table).insert(data).execute()
    return result.data[0] if result.data else {}


def _update(table: str, data: dict, id_val, id_col: str = "id") -> dict:
    result = supabase_client.table(table).update(data).eq(id_col, id_val).execute()
    return result.data[0] if result.data else {}


def _delete(table: str, id_val, id_col: str = "id"):
    supabase_client.table(table).delete().eq(id_col, id_val).execute()


def _delete_in(table: str, col: str, values: list):
    if not values:
        return
    supabase_client.table(table).delete().in_(col, values).execute()


def upload_to_supabase(local_path: Path, remote_path: str) -> str:
    if not supabase_client:
        return str(local_path)
    try:
        content_type = mimetypes.guess_type(str(local_path))[0] or "audio/wav"
        with open(local_path, "rb") as f:
            supabase_client.storage.from_(SUPABASE_BUCKET).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": content_type, "upsert": "true"},
            )
        url = supabase_client.storage.from_(SUPABASE_BUCKET).get_public_url(remote_path)
        print(f"✅ Uploaded to Supabase Storage: {url}")
        return url
    except Exception as _e:
        print(f"⚠️  Supabase upload failed: {_e} — keeping local path")
        return str(local_path)


# ── Python path ──────────────────────────────────────────────────────────────

CURRENT_DIR = Path(__file__).parent.absolute()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ── ML module status ─────────────────────────────────────────────────────────

modules_status = {
    "whisper": False,
    "mfcc": False,
    "emotion_classifier": False,
    "state_classifier": False,
    "recommendation_engine": False,
}

# ── App ───────────────────────────────────────────────────────────────────────

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

# ── Directories ───────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

for _d in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, OUTPUT_DIR / "recommendations"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── ML model instances ────────────────────────────────────────────────────────

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
    if input_path.suffix.lower() == ".wav":
        return input_path
    wav_path = input_path.with_suffix(".wav")
    for fmt in ("ogg", "webm", None):
        try:
            audio = AudioSegment.from_file(str(input_path), format=fmt) if fmt else AudioSegment.from_file(str(input_path))
            audio.export(str(wav_path), format="wav")
            print(f"✅ WAV OK ({fmt or 'auto'})")
            return wav_path
        except Exception as e:
            print(f"⚠️ {fmt or 'auto'} failed: {e}")
    return input_path


def ensure_wav(input_path: Path) -> Path:
    return convert_to_wav(input_path)


def initialize_models():
    global feature_extractor, emotion_classifier, state_classifier, recommendation_engine
    try:
        from mfcc_feature_extraction import MFCCFeatureExtractor as _MFCC
        feature_extractor = _MFCC()
        modules_status["mfcc"] = True
        print("✅ mfcc loaded")
    except Exception as e:
        print(f"❌ mfcc failed: {e}")
    try:
        from ml_classifier import EmotionClassifier as _EC
        emotion_classifier = _EC()
        emotion_classifier.load_all_models(str(MODELS_DIR))
        modules_status["emotion_classifier"] = True
        print("✅ emotion_classifier loaded with SVM, RF, KNN")
    except Exception as e:
        print(f"❌ emotion_classifier failed: {e}")
    try:
        from emotional_state_classifier import EmotionalStateClassifier as _ESC
        state_classifier = _ESC()
        modules_status["state_classifier"] = True
        print("✅ state_classifier loaded")
    except Exception as e:
        print(f"❌ state_classifier failed: {e}")
    try:
        from csr_emotion_recommendations import CSREmotionClassifier as _CEC
        recommendation_engine = _CEC()
        modules_status["recommendation_engine"] = True
        print("✅ recommendation_engine loaded")
    except Exception as e:
        print(f"❌ recommendation_engine failed: {e}")


@app.on_event("startup")
async def startup_event():
    global transcriber
    print("🚀 Loading ALL models at startup...")
    try:
        from whisper_asr_module import CSRCallTranscriber
        transcriber = CSRCallTranscriber(model_size="base", language="en")
        modules_status["whisper"] = True
        print("✅ Whisper loaded")
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
    initialize_models()
    print("🔥 ALL MODELS READY")


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def normalize_model_prediction(prediction: dict) -> dict:
    if not prediction or not isinstance(prediction, dict):
        return {"emotion": "neutral", "confidence": 0.0, "all_probabilities": {}}
    return {
        "emotion": str(prediction.get("emotion") or prediction.get("predicted_emotion") or "neutral").lower(),
        "confidence": float(prediction.get("confidence", 0) or 0),
        "all_probabilities": prediction.get("all_probabilities", {}) or {},
    }


def safe_predict_model(classifier, method_name: str, features):
    try:
        if hasattr(classifier, method_name):
            return getattr(classifier, method_name)(features)
    except Exception as e:
        print(f"⚠️ {method_name} failed: {e}")
    return None


def ensemble_predictions(svm_pred, rf_pred, knn_pred):
    weights = {"svm": 0.50, "rf": 0.30, "knn": 0.20}
    combined = {}
    for model_name, pred in [("svm", svm_pred), ("rf", rf_pred), ("knn", knn_pred)]:
        if not pred:
            continue
        for emotion, prob in (pred.get("all_probabilities") or {}).items():
            k = str(emotion).lower()
            combined[k] = combined.get(k, 0.0) + float(prob) * weights[model_name]
    if not combined:
        return svm_pred or rf_pred or knn_pred or {"emotion": "neutral", "confidence": 0.0, "all_probabilities": {}}
    combined = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    final_emotion = next(iter(combined))
    return {"emotion": final_emotion, "confidence": float(combined[final_emotion]), "all_probabilities": combined, "ensemble_method": "weighted", "weights": weights}


def clean_for_json(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [clean_for_json(v) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.generic):
            return clean_for_json(value.item())
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def compute_risk_level(prediction_dict: dict) -> str:
    emotion = str(prediction_dict.get("emotion", "neutral")).lower()
    confidence = float(prediction_dict.get("confidence", 0.0))
    if emotion == "angry":
        return "Critical" if confidence >= 0.60 else "High"
    elif emotion == "frustrated":
        return "High" if confidence >= 0.60 else "Medium"
    elif emotion == "sad":
        return "Medium"
    return "Low"


# ── ML endpoints ──────────────────────────────────────────────────────────────

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
    if not transcriber:
        raise HTTPException(503, detail="Whisper not loaded")

    missing = [k for k, v in {"feature_extractor": feature_extractor, "emotion_classifier": emotion_classifier, "state_classifier": state_classifier}.items() if v is None]
    if missing:
        raise HTTPException(503, detail=f"Service not ready. Missing: {', '.join(missing)}")
    if not file.filename:
        raise HTTPException(400, detail="No file provided")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = UPLOAD_DIR / f"audio_{timestamp}{Path(file.filename).suffix}"

    try:
        save_upload_file(file, temp_file)
        process_path = convert_to_wav(temp_file)

        transcription_result = transcriber.transcribe_call(str(process_path))
        if not transcription_result or "error" in transcription_result:
            raise HTTPException(500, detail=f"Transcription failed: {transcription_result.get('error', 'Unknown')}")

        speaker_info = transcriber.detect_speakers(str(process_path))
        caller_audio_path = Path(speaker_info["caller_audio_path"])
        if not caller_audio_path.exists() or caller_audio_path.stat().st_size == 0:
            raise HTTPException(500, detail=f"Caller audio split missing or empty: {caller_audio_path}")

        if caller_audio_path.suffix.lower() != ".wav":
            caller_audio_wav = convert_to_wav(caller_audio_path)
        else:
            caller_audio_wav = caller_audio_path

        if not caller_audio_wav.exists() or caller_audio_wav.stat().st_size == 0:
            raise HTTPException(500, detail=f"Caller WAV empty after conversion: {caller_audio_wav}")

        speaker_info["caller_audio_path"] = str(caller_audio_wav)

        features = feature_extractor.extract_features(str(caller_audio_wav))
        if features is None:
            raise HTTPException(500, detail="Feature extraction returned None")

        svm_prediction = safe_predict_model(emotion_classifier, "predict_svm", features)
        rf_prediction  = safe_predict_model(emotion_classifier, "predict_rf",  features)
        knn_prediction = safe_predict_model(emotion_classifier, "predict_knn", features)

        missing_models = [n for n, p in [("SVM", svm_prediction), ("RF", rf_prediction), ("KNN", knn_prediction)] if p is None]
        if missing_models:
            raise HTTPException(500, detail=f"Model prediction failed: {', '.join(missing_models)}")

        svm_prediction = clean_for_json(normalize_model_prediction(svm_prediction))
        rf_prediction  = clean_for_json(normalize_model_prediction(rf_prediction))
        knn_prediction = clean_for_json(normalize_model_prediction(knn_prediction))

        prediction = clean_for_json(ensemble_predictions(svm_prediction, rf_prediction, knn_prediction))
        emotional_state = clean_for_json(state_classifier.classify_emotional_state(prediction))

        recommendations = {"available": False, "message": "Recommendation engine not initialized"}
        if recommendation_engine:
            try:
                csr_state = clean_for_json(recommendation_engine.classify_emotional_state(prediction))
                recommendations = clean_for_json(recommendation_engine.generate_recommendation(csr_state))
                recommendations["report"] = recommendation_engine.generate_report(csr_state, recommendations)
                recommendations["saved_files"] = recommendation_engine.save_recommendation(csr_state, recommendations, output_dir=str(OUTPUT_DIR / "recommendations"))
                recommendations = clean_for_json(recommendations)
            except Exception as e:
                recommendations = {"available": False, "error": str(e)}

        risk_level = compute_risk_level(prediction)

        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "transcription": {
                "text": transcription_result.get("text", ""),
                "language": transcription_result.get("language", "en"),
                "segments": clean_for_json(transcription_result.get("segments", [])),
                "duration": float(transcription_result.get("duration", 0) or 0),
            },
            "speaker_detection": {
                "mode": speaker_info.get("mode"),
                "agent_channel": speaker_info.get("agent_channel"),
                "caller_channel": speaker_info.get("caller_channel"),
                "message": speaker_info.get("message"),
            },
            "emotion_analysis": {
                "predicted_emotion": str(prediction.get("emotion", "neutral")).lower(),
                "confidence": float(prediction.get("confidence", 0)),
                "all_probabilities": clean_for_json(prediction.get("all_probabilities", {})),
                "emotional_state": emotional_state,
                "risk_level": risk_level,
            },
            "model_results": {"svm": svm_prediction, "rf": rf_prediction, "knn": knn_prediction},
            "csr_recommendations": recommendations,
        }

        with open(OUTPUT_DIR / f"analysis_{timestamp}.json", "w", encoding="utf-8") as f_out:
            json.dump(result, f_out, indent=2, ensure_ascii=False)

        # Persist to Supabase DB
        if db_available and agent_id:
            try:
                agent = _get_one("agents", agent_id)
                if agent:
                    emotion_data = result["emotion_analysis"]
                    trans_data   = result["transcription"]
                    spk_data     = result["speaker_detection"]
                    rec_data     = result.get("csr_recommendations", {})
                    call_uuid    = str(uuid_lib.uuid4())

                    call_row = _insert("calls", {
                        "uuid": call_uuid,
                        "agent_id": agent["id"],
                        "cluster_id": agent["cluster_id"],
                        "filename": file.filename,
                        "file_path": str(temp_file),
                        "file_size": temp_file.stat().st_size if temp_file.exists() else None,
                        "duration_sec": int(trans_data.get("duration", 0)) or None,
                        "upload_status": "analyzed",
                        "call_date": date.today().isoformat(),
                    })
                    call_id = call_row["id"]
                    result["call_id"] = call_id

                    for model_name, pred in [("SVM", svm_prediction), ("RF", rf_prediction), ("KNN", knn_prediction)]:
                        _insert("call_model_results", {
                            "call_id": call_id,
                            "model_name": model_name,
                            "predicted_emotion": str(pred.get("emotion", "neutral")).lower(),
                            "confidence": float(pred.get("confidence", 0)),
                            "all_probabilities": json.dumps(clean_for_json(pred.get("all_probabilities") or {})),
                        })

                    ar_row = _insert("analysis_results", {
                        "call_id": call_id,
                        "predicted_emotion": emotion_data["predicted_emotion"],
                        "confidence": emotion_data["confidence"],
                        "all_probabilities": json.dumps(emotion_data.get("all_probabilities") or {}),
                        "valence": emotion_data.get("emotional_state", {}).get("valence"),
                        "arousal": emotion_data.get("emotional_state", {}).get("arousal"),
                        "risk_level": risk_level,
                        "transcription_text": trans_data.get("text"),
                        "transcription_lang": trans_data.get("language", "en"),
                        "transcription_duration": trans_data.get("duration"),
                        "transcription_segments": json.dumps(clean_for_json(trans_data.get("segments", []))),
                        "speaker_mode": spk_data.get("mode"),
                        "agent_channel": spk_data.get("agent_channel"),
                        "caller_channel": spk_data.get("caller_channel"),
                    })
                    ar_id = ar_row["id"]

                    if rec_data.get("available") is not False:
                        action_req = rec_data.get("action_required", {})
                        comm = rec_data.get("communication_guidance", {})
                        dd   = rec_data.get("dos_and_donts", {})
                        _insert("csr_recommendations", {
                            "analysis_result_id": ar_id,
                            "action": action_req.get("action", "NONE"),
                            "urgency": action_req.get("urgency", "LOW"),
                            "reason": action_req.get("reason"),
                            "instruction": action_req.get("instruction"),
                            "action_color": action_req.get("color"),
                            "recommended_tone": comm.get("recommended_tone"),
                            "example_phrases": json.dumps(clean_for_json(comm.get("example_phrases") or [])),
                            "do_list": json.dumps(clean_for_json(dd.get("do") or [])),
                            "dont_list": json.dumps(clean_for_json(dd.get("dont") or [])),
                        })

                    new_risk = "Risky" if risk_level in ("Critical", "High") else "Medium" if risk_level == "Medium" else "Safe"
                    _update("agents", {"risk_level": new_risk}, agent["id"])

                    # Upload to Supabase Storage
                    remote_path = f"calls/{call_uuid}/{process_path.name}"
                    storage_url = upload_to_supabase(process_path, remote_path)
                    if storage_url != str(process_path):
                        _update("calls", {"file_path": storage_url}, call_id)

            except Exception as db_err:
                print(f"⚠️ DB persist error: {db_err}")

        cleanup_file(temp_file)
        if process_path != temp_file:
            cleanup_file(process_path)

        return JSONResponse(content=result)

    except HTTPException:
        cleanup_file(temp_file)
        raise
    except Exception as e:
        cleanup_file(temp_file)
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")


@app.post("/transcribe-only")
async def transcribe_only(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
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
            raise HTTPException(500, detail=f"Transcription failed: {transcription_result.get('error', 'Unknown')}")
        speaker_info = transcriber.detect_speakers(str(process_path))
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        return JSONResponse(content={"success": True, "filename": file.filename, "transcription": transcription_result, "speaker_detection": speaker_info})
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


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DeleteConfirmPayload(BaseModel):
    user_id: int
    password: str

class FlagAgentPayload(BaseModel):
    comment: str

class ClusterCreate(BaseModel):
    name: str

class ClusterUpdate(BaseModel):
    name: Optional[str] = None

class AgentCreate(BaseModel):
    cluster_id: int
    name: str
    email: Optional[str] = None
    role: Optional[str] = None

class AgentUpdate(BaseModel):
    cluster_id: Optional[int] = None
    name: Optional[str] = None
    risk_level: Optional[str] = None
    is_active: Optional[bool] = None

class CallUpdate(BaseModel):
    agent_id: Optional[int] = None
    call_date: Optional[date] = None
    upload_status: Optional[str] = None

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


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/roles")
def list_roles():
    return [
        {"value": "admin", "label": "Admin"},
        {"value": "supervisor", "label": "Supervisor"},
        {"value": "agent", "label": "Agent"},
    ]


@app.post("/auth/register", status_code=201)
def register(payload: RegisterPayload):
    _require_db()

    agent = _get_one("agents", payload.agent_id)
    if not agent:
        raise HTTPException(404, detail="Agent not found")

    existing = _get_by("users", {"agent_id": payload.agent_id})
    if existing:
        raise HTTPException(400, detail="An account already exists for this agent")

    if len(payload.password) < 8:
        raise HTTPException(400, detail="Password must be at least 8 characters")

    if payload.role not in {"admin", "supervisor", "agent"}:
        raise HTTPException(400, detail="Invalid role selected")

    user = _insert("users", {
        "uuid": str(uuid_lib.uuid4()),
        "agent_id": payload.agent_id,
        "username": str(payload.agent_id).zfill(4),
        "password_hash": _hash_password(payload.password),
        "role": payload.role,
        "is_active": True,
    })

    return {
        "id": user["id"],
        "agent_id": user["agent_id"],
        "username": user["username"],
        "role": user["role"],
        "agent_name": agent["name"],
        "agent_email": agent.get("email"),
        "is_active": user["is_active"],
        "created_at": user.get("created_at"),
    }


@app.post("/auth/login")
def login(payload: LoginPayload):
    _require_db()

    users = _get_by("users", {"username": payload.username})
    user = users[0] if users else None

    if not user:
        agents = _get_by("agents", {"email": payload.username})
        if agents:
            rows = _get_by("users", {"agent_id": agents[0]["id"]})
            user = rows[0] if rows else None

    if not user or not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, detail="Invalid username or password")

    if not user.get("is_active", True):
        raise HTTPException(403, detail="Account is disabled")

    _update("users", {"last_login_at": datetime.utcnow().isoformat()}, user["id"])

    agent = _get_one("agents", user["agent_id"]) if user.get("agent_id") else None
    cluster = _get_one("clusters", agent["cluster_id"]) if agent else None

    return {
        "id": user["id"],
        "agent_id": user["agent_id"],
        "username": user["username"],
        "agent_name": agent["name"] if agent else None,
        "agent_email": agent.get("email") if agent else None,
        "role": user["role"],
        "cluster_id": agent["cluster_id"] if agent else None,
        "cluster_name": cluster["name"] if cluster else None,
        "is_active": user.get("is_active", True),
        "last_login_at": user.get("last_login_at"),
    }


# ── Users ─────────────────────────────────────────────────────────────────────

@app.get("/users")
def list_users():
    _require_db()
    users = _get_all("users")
    result = []
    for u in users:
        agent = _get_one("agents", u["agent_id"]) if u.get("agent_id") else None
        result.append({
            "id": u["id"],
            "agent_id": u.get("agent_id"),
            "username": u["username"],
            "role": u["role"],
            "is_active": u.get("is_active", True),
            "agent_name": agent["name"] if agent else None,
            "agent_email": agent.get("email") if agent else None,
            "last_login_at": u.get("last_login_at"),
            "created_at": u.get("created_at"),
        })
    return result


@app.get("/users/{user_id}")
def get_user(user_id: int):
    _require_db()
    u = _get_one("users", user_id)
    if not u:
        raise HTTPException(404, detail="User not found")
    agent = _get_one("agents", u["agent_id"]) if u.get("agent_id") else None
    return {
        "id": u["id"],
        "agent_id": u.get("agent_id"),
        "username": u["username"],
        "is_active": u.get("is_active", True),
        "agent_name": agent["name"] if agent else None,
        "agent_email": agent.get("email") if agent else None,
        "last_login_at": u.get("last_login_at"),
        "created_at": u.get("created_at"),
    }


@app.put("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate):
    _require_db()
    u = _get_one("users", user_id)
    if not u:
        raise HTTPException(404, detail="User not found")
    data = payload.model_dump(exclude_none=True)
    if "password" in data:
        data["password_hash"] = _hash_password(data.pop("password"))
    updated = _update("users", data, user_id)
    return {"id": updated["id"], "agent_id": updated.get("agent_id"), "username": updated["username"], "is_active": updated.get("is_active", True)}


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    _require_db()
    if not _get_one("users", user_id):
        raise HTTPException(404, detail="User not found")
    _delete("users", user_id)


# ── Clusters ──────────────────────────────────────────────────────────────────

@app.get("/clusters")
def list_clusters():
    _require_db()
    clusters = _get_all("clusters")
    agents_all = _get_all("agents")
    today = date.today().isoformat()
    result = []
    for c in clusters:
        cid = c["id"]
        agents = [a for a in agents_all if a.get("cluster_id") == cid]
        calls_today = _count("calls", {"cluster_id": cid})
        total_calls = _count("calls", {"cluster_id": cid})
        # separate counts for today
        calls_today_q = supabase_client.table("calls").select("id", count="exact").eq("cluster_id", cid).eq("call_date", today).execute()
        calls_today_count = calls_today_q.count or 0
        total_q = supabase_client.table("calls").select("id", count="exact").eq("cluster_id", cid).execute()
        total_count = total_q.count or 0
        result.append({
            "id": cid,
            "name": c["name"],
            "region": c.get("region", ""),
            "overall_risk": c.get("overall_risk", "Safe"),
            "agent_count": len(agents),
            "risky_agents": sum(1 for a in agents if a.get("risk_level") == "Risky"),
            "medium_agents": sum(1 for a in agents if a.get("risk_level") == "Medium"),
            "safe_agents": sum(1 for a in agents if a.get("risk_level") == "Safe"),
            "calls_today": calls_today_count,
            "total_calls": total_count,
            "created_at": c.get("created_at"),
        })
    return result


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int):
    _require_db()
    c = _get_one("clusters", cluster_id)
    if not c:
        raise HTTPException(404, detail="Cluster not found")
    return c


@app.post("/clusters", status_code=201)
def create_cluster(payload: ClusterCreate):
    _require_db()
    existing = _get_by("clusters", {"name": payload.name})
    if existing:
        raise HTTPException(400, detail="Cluster name already exists")
    return _insert("clusters", {"name": payload.name})


@app.put("/clusters/{cluster_id}")
def update_cluster(cluster_id: int, payload: ClusterUpdate):
    _require_db()
    if not _get_one("clusters", cluster_id):
        raise HTTPException(404, detail="Cluster not found")
    data = payload.model_dump(exclude_none=True)
    return _update("clusters", data, cluster_id)


@app.delete("/clusters/{cluster_id}")
def delete_cluster(cluster_id: int, payload: DeleteConfirmPayload):
    _require_db()
    user = _get_one("users", payload.user_id)
    if not user or not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, detail="Invalid password")

    c = _get_one("clusters", cluster_id)
    if not c:
        raise HTTPException(404, detail="Cluster not found")

    call_rows = supabase_client.table("calls").select("id").eq("cluster_id", cluster_id).execute().data or []
    call_ids = [r["id"] for r in call_rows]

    if call_ids:
        ar_rows = supabase_client.table("analysis_results").select("id").in_("call_id", call_ids).execute().data or []
        ar_ids = [r["id"] for r in ar_rows]
        if ar_ids:
            _delete_in("csr_recommendations", "analysis_result_id", ar_ids)
        _delete_in("analysis_results", "call_id", call_ids)
        _delete_in("escalations", "call_id", call_ids)
        _delete_in("call_model_results", "call_id", call_ids)
        _delete_in("calls", "id", call_ids)

    # Detach agents from cluster
    agent_rows = supabase_client.table("agents").select("id").eq("cluster_id", cluster_id).execute().data or []
    for a in agent_rows:
        _update("agents", {"cluster_id": None}, a["id"])

    _delete("clusters", cluster_id)
    return {"message": "Cluster deleted successfully"}


# ── Agents ────────────────────────────────────────────────────────────────────

@app.get("/agents")
def list_agents(cluster_id: Optional[int] = None):
    _require_db()
    today = date.today().isoformat()
    agents = _get_all("agents") if not cluster_id else _get_by("agents", {"cluster_id": cluster_id})
    clusters_map = {c["id"]: c for c in _get_all("clusters")}
    result = []
    for a in agents:
        calls_today_q = supabase_client.table("calls").select("id", count="exact").eq("agent_id", a["id"]).eq("call_date", today).execute()
        total_q = supabase_client.table("calls").select("id", count="exact").eq("agent_id", a["id"]).execute()
        cluster = clusters_map.get(a.get("cluster_id", 0))
        result.append({
            "id": a["id"],
            "name": a["name"],
            "email": a.get("email", ""),
            "role": a.get("role", "CSR"),
            "risk_level": a.get("risk_level", "Safe"),
            "is_active": a.get("is_active", True),
            "cluster_id": a.get("cluster_id"),
            "cluster_name": cluster["name"] if cluster else None,
            "calls_today": calls_today_q.count or 0,
            "total_calls": total_q.count or 0,
            "created_at": a.get("created_at"),
        })
    return result


@app.get("/agents/flagged")
def list_flagged_agents():
    _require_db()
    today = date.today().isoformat()
    agents = supabase_client.table("agents").select("*").eq("risk_level", "Risky").execute().data or []
    clusters_map = {c["id"]: c for c in _get_all("clusters")}
    result = []
    for a in agents:
        logs = supabase_client.table("agent_audit_logs").select("*").eq("agent_id", a["id"]).eq("action", "FLAGGED").order("created_at", desc=True).limit(1).execute().data or []
        latest_log = logs[0] if logs else None
        calls_today_q = supabase_client.table("calls").select("id", count="exact").eq("agent_id", a["id"]).eq("call_date", today).execute()
        total_q = supabase_client.table("calls").select("id", count="exact").eq("agent_id", a["id"]).execute()
        cluster = clusters_map.get(a.get("cluster_id", 0))
        result.append({
            "id": a["id"],
            "name": a["name"],
            "email": a.get("email", ""),
            "role": a.get("role", "CSR"),
            "risk_level": a.get("risk_level", "Safe"),
            "is_active": a.get("is_active", True),
            "cluster_id": a.get("cluster_id"),
            "cluster_name": cluster["name"] if cluster else None,
            "calls_today": calls_today_q.count or 0,
            "total_calls": total_q.count or 0,
            "flag_comment": latest_log["comment"] if latest_log else None,
            "flagged_at": latest_log["created_at"] if latest_log else None,
        })
    return result


@app.get("/agents/{agent_id}")
def get_agent(agent_id: int):
    _require_db()
    a = _get_one("agents", agent_id)
    if not a:
        raise HTTPException(404, detail="Agent not found")
    return a


@app.patch("/agents/{agent_id}/flag")
def flag_agent(agent_id: int, payload: FlagAgentPayload):
    _require_db()
    agent = _get_one("agents", agent_id)
    if not agent:
        raise HTTPException(404, detail="Agent not found")
    if not payload.comment or not payload.comment.strip():
        raise HTTPException(400, detail="Comment is required")

    _update("agents", {"risk_level": "Risky"}, agent_id)
    _insert("agent_audit_logs", {
        "agent_id": agent_id,
        "action": "FLAGGED",
        "comment": payload.comment.strip(),
    })

    return {"message": "Agent flagged successfully", "agent_id": agent_id, "name": agent["name"], "risk_level": "Risky"}


@app.patch("/agents/{agent_id}/reset-risk", status_code=200)
def reset_agent_risk(agent_id: int):
    _require_db()
    a = _get_one("agents", agent_id)
    if not a:
        raise HTTPException(404, detail="Agent not found")
    if a.get("risk_level") != "Risky":
        raise HTTPException(400, detail="Agent is not flagged as Risky")
    updated = _update("agents", {"risk_level": "Safe"}, agent_id)
    return {"id": updated["id"], "name": updated["name"], "risk_level": updated["risk_level"]}


@app.post("/agents", status_code=201)
def create_agent(payload: AgentCreate):
    _require_db()
    if not _get_one("clusters", payload.cluster_id):
        raise HTTPException(404, detail="Cluster not found")
    data = payload.model_dump(exclude_none=True)
    if "email" not in data:
        data["email"] = f"agent-{uuid_lib.uuid4().hex[:12]}@placeholder.internal"
    try:
        return _insert("agents", data)
    except _PostgRESTAPIError as e:
        code = getattr(e, "code", "") or ""
        details = str(getattr(e, "details", "") or "")
        message = str(getattr(e, "message", "") or "")
        if code == "23505":
            if "agents_email_key" in details or "agents_email_key" in message:
                raise HTTPException(409, detail="An agent with this email already exists.")
            if "agents_pkey" in details or "agents_pkey" in message:
                raise HTTPException(
                    409,
                    detail="ID sequence conflict. Run this in Supabase SQL editor: "
                           "SELECT setval(pg_get_serial_sequence('agents','id'), MAX(id)) FROM agents;",
                )
            raise HTTPException(409, detail="Duplicate entry — agent already exists.")
        raise


@app.put("/agents/{agent_id}")
def update_agent(agent_id: int, payload: AgentUpdate):
    _require_db()
    if not _get_one("agents", agent_id):
        raise HTTPException(404, detail="Agent not found")
    return _update("agents", payload.model_dump(exclude_none=True), agent_id)


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: int):
    _require_db()
    if not _get_one("agents", agent_id):
        raise HTTPException(404, detail="Agent not found")
    _delete("agents", agent_id)


# ── Calls ─────────────────────────────────────────────────────────────────────

@app.get("/calls")
def list_calls(cluster_id: Optional[int] = None, agent_id: Optional[int] = None):
    _require_db()
    q = supabase_client.table("calls").select("*")
    if cluster_id:
        q = q.eq("cluster_id", cluster_id)
    if agent_id:
        q = q.eq("agent_id", agent_id)
    calls = q.execute().data or []

    agents_map   = {a["id"]: a for a in _get_all("agents")}
    clusters_map = {c["id"]: c for c in _get_all("clusters")}

    result = []
    for c in calls:
        ar_rows = _get_by("analysis_results", {"call_id": c["id"]})
        analysis = ar_rows[0] if ar_rows else None

        model_rows = supabase_client.table("call_model_results").select("*").eq("call_id", c["id"]).execute().data or []
        model_results = {m["model_name"].lower(): {"emotion": m["predicted_emotion"], "confidence": float(m["confidence"])} for m in model_rows}

        rec_rows = _get_by("csr_recommendations", {"analysis_result_id": analysis["id"]}) if analysis else []
        rec = rec_rows[0] if rec_rows else None

        agent   = agents_map.get(c.get("agent_id"))
        cluster = clusters_map.get(c.get("cluster_id"))

        result.append({
            "id": c["id"],
            "filename": c.get("filename"),
            "duration_sec": c.get("duration_sec"),
            "call_date": c.get("call_date"),
            "upload_status": c.get("upload_status"),
            "agent": {"id": agent["id"], "name": agent["name"], "email": agent.get("email")} if agent else None,
            "cluster": {"id": cluster["id"], "name": cluster["name"]} if cluster else None,
            "analysis": {
                "predicted_emotion": analysis["predicted_emotion"],
                "confidence": float(analysis["confidence"]),
                "risk_level": analysis["risk_level"],
                "transcription_text": analysis.get("transcription_text"),
                "transcription_segments": json.loads(analysis.get("transcription_segments") or "[]"),
                "model_results": model_results,
            } if analysis else None,
            "recommendation": {
                "action": rec["action"] if rec else None,
                "urgency": rec["urgency"] if rec else None,
                "reason": rec.get("reason") if rec else None,
                "instruction": rec.get("instruction") if rec else None,
                "recommended_tone": rec.get("recommended_tone") if rec else None,
            } if analysis else None,
        })
    return result


@app.get("/calls/{call_id}/audio")
def stream_audio(call_id: int):
    _require_db()
    call = _get_one("calls", call_id)
    if not call:
        raise HTTPException(404, detail="Call not found")
    if call.get("file_path") and call["file_path"].startswith("http"):
        return RedirectResponse(url=call["file_path"])
    path = Path(call.get("file_path", ""))
    if not path.exists():
        raise HTTPException(404, detail="Audio file not found on disk")
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(path=str(path), media_type=mime or "audio/mpeg", filename=call.get("filename"))


@app.get("/calls/{call_id}/model-results")
def get_call_model_results(call_id: int):
    _require_db()
    if not _get_one("calls", call_id):
        raise HTTPException(404, detail="Call not found")
    rows = supabase_client.table("call_model_results").select("*").eq("call_id", call_id).order("id").execute().data or []
    return [
        {
            "id": r["id"],
            "call_id": r["call_id"],
            "model_name": r["model_name"],
            "predicted_emotion": r["predicted_emotion"],
            "confidence": float(r["confidence"]),
            "all_probabilities": json.loads(r.get("all_probabilities") or "{}"),
            "created_at": r.get("created_at"),
        }
        for r in rows
    ]


@app.put("/calls/{call_id}")
def update_call(call_id: int, payload: CallUpdate):
    _require_db()
    if not _get_one("calls", call_id):
        raise HTTPException(404, detail="Call not found")
    data = payload.model_dump(exclude_none=True)
    if "call_date" in data and isinstance(data["call_date"], date):
        data["call_date"] = data["call_date"].isoformat()
    return _update("calls", data, call_id)


@app.delete("/calls/{call_id}")
def delete_call(call_id: int, payload: DeleteConfirmPayload):
    _require_db()
    user = _get_one("users", payload.user_id)
    if not user or not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, detail="Invalid password")

    if not _get_one("calls", call_id):
        raise HTTPException(404, detail="Call not found")

    ar_rows = _get_by("analysis_results", {"call_id": call_id})
    if ar_rows:
        ar_ids = [r["id"] for r in ar_rows]
        _delete_in("csr_recommendations", "analysis_result_id", ar_ids)
        _delete_in("analysis_results", "call_id", [call_id])

    _delete_in("escalations", "call_id", [call_id])
    _delete_in("call_model_results", "call_id", [call_id])
    _delete("calls", call_id)
    return {"message": "Call deleted successfully"}


# ── Reports ───────────────────────────────────────────────────────────────────

@app.get("/reports/summary")
def reports_summary(cluster_id: Optional[int] = None):
    _require_db()

    if cluster_id:
        call_ids_q = supabase_client.table("calls").select("id").eq("cluster_id", cluster_id).execute().data or []
        call_ids = [r["id"] for r in call_ids_q]
        total_calls = len(call_ids)
        analyzed = len(supabase_client.table("analysis_results").select("id").in_("call_id", call_ids).execute().data or []) if call_ids else 0
        escalations = len(supabase_client.table("escalations").select("id").in_("call_id", call_ids).execute().data or []) if call_ids else 0
    else:
        total_calls  = supabase_client.table("calls").select("id", count="exact").execute().count or 0
        analyzed     = supabase_client.table("analysis_results").select("id", count="exact").execute().count or 0
        escalations  = supabase_client.table("escalations").select("id", count="exact").execute().count or 0

    total_agents = supabase_client.table("agents").select("id", count="exact").execute().count or 0
    risky_agents = supabase_client.table("agents").select("id", count="exact").eq("risk_level", "Risky").execute().count or 0

    return {
        "total_calls": total_calls,
        "analyzed_calls": analyzed,
        "escalations": escalations,
        "total_agents": total_agents,
        "risky_agents": risky_agents,
    }


def _build_7day_pivot():
    start = date.today() - timedelta(days=6)
    pivot = {}
    for i in range(7):
        d = start + timedelta(days=i)
        label = d.strftime("%a")
        pivot[label] = {"day": label, "date": d.isoformat(), "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "total": 0}
    return pivot, start


@app.get("/reports/flagged-agent-daily-trend")
def reports_flagged_agent_daily_trend():
    _require_db()
    pivot, start = _build_7day_pivot()

    flagged = supabase_client.table("agents").select("id").eq("risk_level", "Risky").execute().data or []
    flagged_ids = [a["id"] for a in flagged]
    if not flagged_ids:
        return list(pivot.values())

    calls = supabase_client.table("calls").select("id,call_date,agent_id").in_("agent_id", flagged_ids).gte("call_date", start.isoformat()).execute().data or []
    if not calls:
        return list(pivot.values())

    call_ids = [c["id"] for c in calls]
    call_date_map = {c["id"]: c["call_date"] for c in calls}

    ars = supabase_client.table("analysis_results").select("call_id,risk_level").in_("call_id", call_ids).execute().data or []
    for ar in ars:
        call_date_str = call_date_map.get(ar["call_id"], "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = d.strftime("%a")
        if label in pivot:
            rl = ar.get("risk_level", "Low")
            pivot[label][rl] = pivot[label].get(rl, 0) + 1
            pivot[label]["total"] += 1

    return list(pivot.values())


@app.get("/reports/agent-daily-trend")
def reports_agent_daily_trend(agent_id: Optional[int] = None, cluster_id: Optional[int] = None):
    _require_db()
    pivot, start = _build_7day_pivot()

    q = supabase_client.table("calls").select("id,call_date").gte("call_date", start.isoformat())
    if agent_id:
        q = q.eq("agent_id", agent_id)
    if cluster_id:
        q = q.eq("cluster_id", cluster_id)
    calls = q.execute().data or []
    if not calls:
        return list(pivot.values())

    call_ids = [c["id"] for c in calls]
    call_date_map = {c["id"]: c["call_date"] for c in calls}

    ars = supabase_client.table("analysis_results").select("call_id,risk_level").in_("call_id", call_ids).execute().data or []
    for ar in ars:
        call_date_str = call_date_map.get(ar["call_id"], "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = d.strftime("%a")
        if label in pivot:
            rl = ar.get("risk_level", "Low")
            pivot[label][rl] = pivot[label].get(rl, 0) + 1
            pivot[label]["total"] += 1

    return list(pivot.values())


@app.get("/reports/agent-past-trend")
def reports_agent_past_trend(agent_id: Optional[int] = None, cluster_id: Optional[int] = None):
    _require_db()
    start = date.today() - timedelta(days=180)

    q = supabase_client.table("calls").select("id,call_date").gte("call_date", start.isoformat())
    if agent_id:
        q = q.eq("agent_id", agent_id)
    if cluster_id:
        q = q.eq("cluster_id", cluster_id)
    calls = q.execute().data or []

    call_ids = [c["id"] for c in calls]
    call_date_map = {c["id"]: c["call_date"] for c in calls}

    if not call_ids:
        return []

    ars = supabase_client.table("analysis_results").select("call_id,risk_level").in_("call_id", call_ids).execute().data or []

    pivot: dict = {}
    for ar in ars:
        call_date_str = call_date_map.get(ar["call_id"], "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = f"{d.year}-{d.month:02d}"
        if label not in pivot:
            pivot[label] = {"month": label, "Critical": 0, "High": 0, "Medium": 0, "Low": 0, "total": 0}
        rl = ar.get("risk_level", "Low")
        pivot[label][rl] = pivot[label].get(rl, 0) + 1
        pivot[label]["total"] += 1

    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/emotion-distribution")
def reports_emotion_distribution(cluster_id: Optional[int] = None):
    _require_db()
    if cluster_id:
        call_ids_q = supabase_client.table("calls").select("id").eq("cluster_id", cluster_id).execute().data or []
        call_ids = [r["id"] for r in call_ids_q]
        ars = supabase_client.table("analysis_results").select("predicted_emotion").in_("call_id", call_ids).execute().data or [] if call_ids else []
    else:
        ars = supabase_client.table("analysis_results").select("predicted_emotion").execute().data or []

    counts: dict = {}
    for ar in ars:
        e = ar.get("predicted_emotion", "neutral")
        counts[e] = counts.get(e, 0) + 1
    return [{"emotion": e, "count": c} for e, c in counts.items()]


@app.get("/reports/emotion-trend")
def reports_emotion_trend(cluster_id: Optional[int] = None):
    _require_db()
    cutoff = date.today() - timedelta(days=365)

    q = supabase_client.table("calls").select("id,call_date").gte("call_date", cutoff.isoformat())
    if cluster_id:
        q = q.eq("cluster_id", cluster_id)
    calls = q.execute().data or []

    call_ids = [c["id"] for c in calls]
    call_date_map = {c["id"]: c["call_date"] for c in calls}

    if not call_ids:
        return []

    ars = supabase_client.table("analysis_results").select("call_id,predicted_emotion").in_("call_id", call_ids).execute().data or []

    pivot: dict = {}
    for ar in ars:
        call_date_str = call_date_map.get(ar["call_id"], "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = f"{d.year}-{d.month:02d}"
        pivot.setdefault(label, {"month": label})[ar.get("predicted_emotion", "neutral")] = pivot.get(label, {}).get(ar.get("predicted_emotion", "neutral"), 0) + 1

    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/risk-trend")
def reports_risk_trend(cluster_id: Optional[int] = None):
    _require_db()
    cutoff = date.today() - timedelta(days=365)

    q = supabase_client.table("calls").select("id,call_date").gte("call_date", cutoff.isoformat())
    if cluster_id:
        q = q.eq("cluster_id", cluster_id)
    calls = q.execute().data or []

    call_ids = [c["id"] for c in calls]
    call_date_map = {c["id"]: c["call_date"] for c in calls}

    if not call_ids:
        return []

    ars = supabase_client.table("analysis_results").select("call_id,risk_level").in_("call_id", call_ids).execute().data or []

    pivot: dict = {}
    for ar in ars:
        call_date_str = call_date_map.get(ar["call_id"], "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = f"{d.year}-{d.month:02d}"
        pivot.setdefault(label, {"month": label})[ar.get("risk_level", "Low")] = pivot.get(label, {}).get(ar.get("risk_level", "Low"), 0) + 1

    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/call-volume")
def reports_call_volume():
    _require_db()
    cutoff = date.today() - timedelta(days=365)
    calls = supabase_client.table("calls").select("call_date,cluster_id").gte("call_date", cutoff.isoformat()).execute().data or []
    clusters_map = {c["id"]: c["name"] for c in _get_all("clusters")}

    pivot: dict = {}
    for c in calls:
        call_date_str = c.get("call_date", "")
        if not call_date_str:
            continue
        d = date.fromisoformat(str(call_date_str)[:10])
        label = f"{d.year}-{d.month:02d}"
        cluster_name = clusters_map.get(c.get("cluster_id"), "Unknown")
        pivot.setdefault(label, {"month": label})[cluster_name] = pivot.get(label, {}).get(cluster_name, 0) + 1

    return sorted(pivot.values(), key=lambda x: x["month"])


@app.get("/reports/agent-risk-scores")
def reports_agent_risk_scores(cluster_id: Optional[int] = None):
    _require_db()
    agents = _get_all("agents") if not cluster_id else _get_by("agents", {"cluster_id": cluster_id})
    clusters_map = {c["id"]: c["name"] for c in _get_all("clusters")}

    result = []
    for a in agents:
        call_rows = supabase_client.table("calls").select("id").eq("agent_id", a["id"]).execute().data or []
        call_ids = [c["id"] for c in call_rows]

        rm = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        if call_ids:
            ars = supabase_client.table("analysis_results").select("risk_level").in_("call_id", call_ids).execute().data or []
            for ar in ars:
                rl = ar.get("risk_level", "Low")
                rm[rl] = rm.get(rl, 0) + 1

        total = sum(rm.values()) or 1
        result.append({
            "agent_id": a["id"],
            "agent_name": a["name"],
            "cluster": clusters_map.get(a.get("cluster_id", 0)),
            "risk_level": a.get("risk_level", "Safe"),
            "critical": rm["Critical"],
            "high": rm["High"],
            "medium": rm["Medium"],
            "low": rm["Low"],
            "total_calls": total,
            "risk_score": round((rm["Critical"] * 4 + rm["High"] * 3 + rm["Medium"] * 2 + rm["Low"]) / total * 25, 1),
        })
    return sorted(result, key=lambda x: -x["risk_score"])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")
