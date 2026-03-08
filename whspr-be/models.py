"""
SQLAlchemy ORM models — PostgreSQL / Supabase
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Date,
    BigInteger, SmallInteger, ForeignKey, Boolean, Numeric
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import uuid as _uuid


# ── Re-usable ENUM types ──────────────────────────────────────────────────────

user_role_enum     = ENUM("agent", "supervisor",                                        name="user_role",      create_type=False)
risk_level_enum    = ENUM("Safe", "Medium", "Risky",                                    name="risk_level",     create_type=False)
upload_status_enum = ENUM("pending", "processing", "analyzed", "failed",                name="upload_status",  create_type=False)
emotion_enum       = ENUM("angry", "frustrated", "sad", "neutral", "happy", "satisfied",name="emotion",        create_type=False)
valence_enum       = ENUM("positive", "negative", "neutral",                            name="valence",        create_type=False)
arousal_enum       = ENUM("high", "low", "neutral",                                     name="arousal",        create_type=False)
analysis_risk_enum = ENUM("Critical", "High", "Medium", "Low",                          name="analysis_risk",  create_type=False)
action_enum        = ENUM("ESCALATE", "REST", "MONITOR", "NONE",                        name="csr_action",     create_type=False)
urgency_enum       = ENUM("IMMEDIATE", "HIGH", "MEDIUM", "LOW",                         name="csr_urgency",    create_type=False)
action_color_enum  = ENUM("red", "orange", "yellow", "green",                           name="action_color",   create_type=False)


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    uuid          = Column(UUID(as_uuid=True), unique=True, nullable=False, default=_uuid.uuid4)
    name          = Column(String(100), nullable=False)
    email         = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(user_role_enum, nullable=False, default="agent")
    is_active     = Column(Boolean, nullable=False, default=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Cluster(Base):
    __tablename__ = "clusters"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    name         = Column(String(100), unique=True, nullable=False)
    region       = Column(String(100), nullable=False)
    overall_risk = Column(risk_level_enum, nullable=False, default="Safe")
    created_by   = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    agents = relationship("Agent", back_populates="cluster")


class Agent(Base):
    __tablename__ = "agents"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id", ondelete="RESTRICT"), nullable=False)
    name       = Column(String(100), nullable=False)
    email      = Column(String(150), unique=True, nullable=False)
    role       = Column(String(80), nullable=False, default="CSR")
    risk_level = Column(risk_level_enum, nullable=False, default="Safe")
    is_active  = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    cluster = relationship("Cluster", back_populates="agents")
    calls   = relationship("Call", back_populates="agent")


class Call(Base):
    __tablename__ = "calls"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    uuid          = Column(UUID(as_uuid=True), unique=True, nullable=False, default=_uuid.uuid4)
    agent_id      = Column(Integer, ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False)
    cluster_id    = Column(Integer, ForeignKey("clusters.id", ondelete="RESTRICT"), nullable=False)
    filename      = Column(String(255), nullable=False)
    file_path     = Column(String(512), nullable=True)
    file_size     = Column(BigInteger, nullable=True)
    duration_sec  = Column(SmallInteger, nullable=True)
    upload_status = Column(upload_status_enum, nullable=False, default="pending")
    uploaded_by   = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    call_date     = Column(Date, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    agent           = relationship("Agent", back_populates="calls")
    cluster         = relationship("Cluster")
    analysis_result = relationship("AnalysisResult", back_populates="call", uselist=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    call_id                = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), unique=True, nullable=False)
    predicted_emotion      = Column(emotion_enum, nullable=False)
    confidence             = Column(Numeric(5, 4), nullable=False)
    all_probabilities      = Column(JSONB, nullable=True)
    valence                = Column(valence_enum, nullable=True)
    arousal                = Column(arousal_enum, nullable=True)
    risk_level             = Column(analysis_risk_enum, nullable=False, default="Low")
    transcription_text     = Column(Text, nullable=True)
    transcription_lang     = Column(String(10), nullable=True, default="en")
    transcription_duration = Column(Numeric(8, 2), nullable=True)
    speaker_mode           = Column(String(50), nullable=True)
    agent_channel          = Column(String(10), nullable=True)
    caller_channel         = Column(String(10), nullable=True)
    analyzed_at            = Column(DateTime(timezone=True), server_default=func.now())

    call           = relationship("Call", back_populates="analysis_result")
    recommendation = relationship("CSRRecommendation", back_populates="analysis_result", uselist=False)


class CSRRecommendation(Base):
    __tablename__ = "csr_recommendations"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)
    action             = Column(action_enum, nullable=False, default="NONE")
    urgency            = Column(urgency_enum, nullable=False, default="LOW")
    reason             = Column(Text, nullable=True)
    instruction        = Column(Text, nullable=True)
    action_color       = Column(action_color_enum, nullable=True)
    recommended_tone   = Column(Text, nullable=True)
    example_phrases    = Column(JSONB, nullable=True)
    do_list            = Column(JSONB, nullable=True)
    dont_list          = Column(JSONB, nullable=True)
    created_at         = Column(DateTime(timezone=True), server_default=func.now())

    analysis_result = relationship("AnalysisResult", back_populates="recommendation")


class Escalation(Base):
    __tablename__ = "escalations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    call_id      = Column(Integer, ForeignKey("calls.id", ondelete="CASCADE"), nullable=False)
    agent_id     = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    escalated_to = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reason       = Column(Text, nullable=True)
    resolved     = Column(Boolean, nullable=False, default=False)
    resolved_at  = Column(DateTime(timezone=True), nullable=True)
    resolved_by  = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    notes        = Column(Text, nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AgentDailyStat(Base):
    __tablename__ = "agent_daily_stats"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    agent_id         = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    stat_date        = Column(Date, nullable=False)
    calls_count      = Column(SmallInteger, nullable=False, default=0)
    angry_count      = Column(SmallInteger, nullable=False, default=0)
    frustrated_count = Column(SmallInteger, nullable=False, default=0)
    neutral_count    = Column(SmallInteger, nullable=False, default=0)
    happy_count      = Column(SmallInteger, nullable=False, default=0)
    sad_count        = Column(SmallInteger, nullable=False, default=0)
    escalations      = Column(SmallInteger, nullable=False, default=0)
    avg_risk_score   = Column(Numeric(5, 2), nullable=False, default=0.00)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    updated_at       = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
