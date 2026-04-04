"""
SQLAlchemy ORM models — MariaDB/MySQL
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Date,
    BigInteger, SmallInteger, ForeignKey, Boolean, Numeric
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime
import uuid as uuid_lib


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False,
                  default=lambda: str(uuid_lib.uuid4()))
    agent_id = Column(Integer, ForeignKey(
        "agents.id", ondelete="CASCADE",
        use_alter=True, name="fk_users_agent_id"),
        unique=True, nullable=False)
    role = Column(String, nullable=False, default="agent")
    username = Column(String(10), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow, nullable=False)

    agent = relationship("Agent", back_populates="user", uselist=False,
                         foreign_keys=[agent_id])


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    region = Column(String(100), nullable=False)
    overall_risk = Column(String(20), nullable=False, default="Safe")
    created_by = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL",
        use_alter=True, name="fk_clusters_created_by"),
        nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    agents = relationship("Agent", back_populates="cluster")


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey(
        "clusters.id", ondelete="RESTRICT"), nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    role = Column(String(80), nullable=False, default="CSR")
    risk_level = Column(String(20), nullable=False, default="Safe")
    is_active = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL",
        use_alter=True, name="fk_agents_created_by"),
        nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    cluster = relationship("Cluster", back_populates="agents")
    calls = relationship("Call", back_populates="agent")
    user = relationship("User", back_populates="agent", uselist=False,
                        foreign_keys="User.agent_id")


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False,
                  default=lambda: str(uuid_lib.uuid4()))
    agent_id = Column(Integer, ForeignKey(
        "agents.id", ondelete="RESTRICT"), nullable=False)
    cluster_id = Column(Integer, ForeignKey(
        "clusters.id", ondelete="RESTRICT"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    duration_sec = Column(SmallInteger, nullable=True)
    upload_status = Column(String(20), nullable=False, default="pending")
    uploaded_by = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL",
        use_alter=True, name="fk_calls_uploaded_by"),
        nullable=True)
    call_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())

    agent = relationship("Agent", back_populates="calls")
    cluster = relationship("Cluster")
    analysis_result = relationship("AnalysisResult", back_populates="call",
                                   uselist=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey(
        "calls.id", ondelete="CASCADE"),
        unique=True, nullable=False)
    predicted_emotion = Column(String(30), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    all_probabilities = Column(Text, nullable=True)   # stored as JSON string
    valence = Column(String(20), nullable=True)
    arousal = Column(String(20), nullable=True)
    risk_level = Column(String(20), nullable=False, default="Low")
    transcription_text = Column(Text, nullable=True)
    transcription_lang = Column(String(10), nullable=True, default="en")
    transcription_duration = Column(Numeric(8, 2), nullable=True)
    speaker_mode = Column(String(50), nullable=True)
    agent_channel = Column(String(10), nullable=True)
    caller_channel = Column(String(10), nullable=True)
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="analysis_result")
    recommendation = relationship("CSRRecommendation",
                                  back_populates="analysis_result", uselist=False)


class CSRRecommendation(Base):
    __tablename__ = "csr_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_result_id = Column(Integer, ForeignKey(
        "analysis_results.id", ondelete="CASCADE"),
        unique=True, nullable=False)
    action = Column(String(20), nullable=False, default="NONE")
    urgency = Column(String(20), nullable=False, default="LOW")
    reason = Column(Text, nullable=True)
    instruction = Column(Text, nullable=True)
    action_color = Column(String(20), nullable=True)
    recommended_tone = Column(Text, nullable=True)
    example_phrases = Column(Text, nullable=True)   # JSON string
    do_list = Column(Text, nullable=True)   # JSON string
    dont_list = Column(Text, nullable=True)   # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis_result = relationship(
        "AnalysisResult", back_populates="recommendation")


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey(
        "calls.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(Integer, ForeignKey(
        "agents.id", ondelete="CASCADE"), nullable=False)
    escalated_to = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL",
        use_alter=True, name="fk_escalations_escalated_to"),
        nullable=True)
    reason = Column(Text, nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(Integer, ForeignKey(
        "users.id", ondelete="SET NULL",
        use_alter=True, name="fk_escalations_resolved_by"),
        nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())


class AgentDailyStat(Base):
    __tablename__ = "agent_daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, ForeignKey(
        "agents.id", ondelete="CASCADE"), nullable=False)
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
    updated_at = Column(DateTime(timezone=True),
                        server_default=func.now(), onupdate=func.now())
