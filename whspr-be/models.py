"""
SQLAlchemy ORM models — PostgreSQL (Supabase)
No database-level foreign key constraints; relationships use explicit joins.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Date, BigInteger, SmallInteger, Boolean, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime
import uuid as uuid_lib


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid_lib.uuid4()))
    agent_id = Column(Integer, unique=True, nullable=False)
    role = Column(String, nullable=False, default="agent")
    username = Column(String(10), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    agent = relationship(
        "Agent",
        primaryjoin="User.agent_id == Agent.id",
        foreign_keys="[User.agent_id]",
        back_populates="user",
        uselist=False,
    )


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    region = Column(String(100), nullable=False, default="")
    overall_risk = Column(String(20), nullable=False, default="Safe")
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    agents = relationship(
        "Agent",
        primaryjoin="Cluster.id == Agent.cluster_id",
        foreign_keys="[Agent.cluster_id]",
        back_populates="cluster",
    )


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, default="")
    role = Column(String(80), nullable=False, default="CSR")
    risk_level = Column(String(20), nullable=False, default="Safe")
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    cluster = relationship(
        "Cluster",
        primaryjoin="Agent.cluster_id == Cluster.id",
        foreign_keys="[Agent.cluster_id]",
        back_populates="agents",
    )
    calls = relationship(
        "Call",
        primaryjoin="Agent.id == Call.agent_id",
        foreign_keys="[Call.agent_id]",
        back_populates="agent",
    )
    user = relationship(
        "User",
        primaryjoin="Agent.id == User.agent_id",
        foreign_keys="[User.agent_id]",
        back_populates="agent",
        uselist=False,
    )


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid_lib.uuid4()))
    agent_id = Column(Integer, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    duration_sec = Column(SmallInteger, nullable=True)
    upload_status = Column(String(20), nullable=False, default="pending")
    uploaded_by = Column(Integer, nullable=True)
    call_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    agent = relationship(
        "Agent",
        primaryjoin="Call.agent_id == Agent.id",
        foreign_keys="[Call.agent_id]",
        back_populates="calls",
    )
    cluster = relationship(
        "Cluster",
        primaryjoin="Call.cluster_id == Cluster.id",
        foreign_keys="[Call.cluster_id]",
    )
    analysis_result = relationship(
        "AnalysisResult",
        primaryjoin="Call.id == AnalysisResult.call_id",
        foreign_keys="[AnalysisResult.call_id]",
        back_populates="call",
        uselist=False,
    )
    model_results = relationship(
        "CallModelResult",
        primaryjoin="Call.id == CallModelResult.call_id",
        foreign_keys="[CallModelResult.call_id]",
        back_populates="call",
        cascade="all, delete-orphan",
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, unique=True, nullable=False)
    predicted_emotion = Column(String(30), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    all_probabilities = Column(Text, nullable=True)
    valence = Column(String(20), nullable=True)
    arousal = Column(String(20), nullable=True)
    risk_level = Column(String(20), nullable=False, default="Low")
    transcription_text = Column(Text, nullable=True)
    transcription_lang = Column(String(10), nullable=True, default="en")
    transcription_duration = Column(Numeric(8, 2), nullable=True)
    transcription_segments = Column(Text, nullable=True)
    speaker_mode = Column(String(50), nullable=True)
    agent_channel = Column(String(10), nullable=True)
    caller_channel = Column(String(10), nullable=True)
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship(
        "Call",
        primaryjoin="AnalysisResult.call_id == Call.id",
        foreign_keys="[AnalysisResult.call_id]",
        back_populates="analysis_result",
    )
    recommendation = relationship(
        "CSRRecommendation",
        primaryjoin="AnalysisResult.id == CSRRecommendation.analysis_result_id",
        foreign_keys="[CSRRecommendation.analysis_result_id]",
        back_populates="analysis_result",
        uselist=False,
    )


class CSRRecommendation(Base):
    __tablename__ = "csr_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_result_id = Column(Integer, unique=True, nullable=False)
    action = Column(String(20), nullable=False, default="NONE")
    urgency = Column(String(20), nullable=False, default="LOW")
    reason = Column(Text, nullable=True)
    instruction = Column(Text, nullable=True)
    action_color = Column(String(20), nullable=True)
    recommended_tone = Column(Text, nullable=True)
    example_phrases = Column(Text, nullable=True)
    do_list = Column(Text, nullable=True)
    dont_list = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis_result = relationship(
        "AnalysisResult",
        primaryjoin="CSRRecommendation.analysis_result_id == AnalysisResult.id",
        foreign_keys="[CSRRecommendation.analysis_result_id]",
        back_populates="recommendation",
    )


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, nullable=False)
    agent_id = Column(Integer, nullable=False)
    escalated_to = Column(Integer, nullable=True)
    reason = Column(Text, nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AgentDailyStat(Base):
    __tablename__ = "agent_daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, nullable=False)
    stat_date = Column(Date, nullable=False)
    calls_count = Column(SmallInteger, nullable=False, default=0)
    angry_count = Column(SmallInteger, nullable=False, default=0)
    frustrated_count = Column(SmallInteger, nullable=False, default=0)
    neutral_count = Column(SmallInteger, nullable=False, default=0)
    happy_count = Column(SmallInteger, nullable=False, default=0)
    sad_count = Column(SmallInteger, nullable=False, default=0)
    escalations = Column(SmallInteger, nullable=False, default=0)
    avg_risk_score = Column(Numeric(5, 2), nullable=False, default=0.00)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CallModelResult(Base):
    __tablename__ = "call_model_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, nullable=False)
    model_name = Column(String(20), nullable=False)
    predicted_emotion = Column(String(30), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    all_probabilities = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship(
        "Call",
        primaryjoin="CallModelResult.call_id == Call.id",
        foreign_keys="[CallModelResult.call_id]",
        back_populates="model_results",
    )


class AgentAuditLog(Base):
    __tablename__ = "agent_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, nullable=False)
    action = Column(String(100), nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    agent = relationship(
        "Agent",
        primaryjoin="AgentAuditLog.agent_id == Agent.id",
        foreign_keys="[AgentAuditLog.agent_id]",
    )
