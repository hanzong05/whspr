-- =============================================================================
-- WHSPR — PostgreSQL Schema (Supabase)
-- =============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

CREATE TYPE user_role      AS ENUM ('agent', 'supervisor');
CREATE TYPE risk_level     AS ENUM ('Safe', 'Medium', 'Risky');
CREATE TYPE upload_status  AS ENUM ('pending', 'processing', 'analyzed', 'failed');
CREATE TYPE emotion        AS ENUM ('angry', 'frustrated', 'sad', 'neutral', 'happy', 'satisfied');
CREATE TYPE valence        AS ENUM ('positive', 'negative', 'neutral');
CREATE TYPE arousal        AS ENUM ('high', 'low', 'neutral');
CREATE TYPE analysis_risk  AS ENUM ('Critical', 'High', 'Medium', 'Low');
CREATE TYPE csr_action     AS ENUM ('ESCALATE', 'REST', 'MONITOR', 'NONE');
CREATE TYPE csr_urgency    AS ENUM ('IMMEDIATE', 'HIGH', 'MEDIUM', 'LOW');
CREATE TYPE action_color   AS ENUM ('red', 'orange', 'yellow', 'green');
CREATE TYPE audit_action   AS ENUM ('CREATE', 'UPDATE', 'DELETE', 'LOGIN', 'LOGOUT', 'EXPORT');

-- =============================================================================
-- updated_at trigger (reused by every table)
-- =============================================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- USERS
-- =============================================================================

CREATE TABLE users (
  id            SERIAL        PRIMARY KEY,
  uuid          UUID          NOT NULL DEFAULT gen_random_uuid(),
  name          VARCHAR(100)  NOT NULL,
  email         VARCHAR(150)  NOT NULL,
  password_hash VARCHAR(255)  NOT NULL,
  role          user_role     NOT NULL DEFAULT 'agent',
  is_active     BOOLEAN       NOT NULL DEFAULT TRUE,
  last_login_at TIMESTAMPTZ,
  created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_users_uuid  UNIQUE (uuid),
  CONSTRAINT uq_users_email UNIQUE (email)
);
CREATE TRIGGER trg_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- CLUSTERS
-- =============================================================================

CREATE TABLE clusters (
  id           SERIAL        PRIMARY KEY,
  name         VARCHAR(100)  NOT NULL,
  region       VARCHAR(100)  NOT NULL,
  overall_risk risk_level    NOT NULL DEFAULT 'Safe',
  created_by   INTEGER       REFERENCES users (id) ON DELETE SET NULL,
  created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_clusters_name UNIQUE (name)
);
CREATE TRIGGER trg_clusters_updated_at BEFORE UPDATE ON clusters FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- AGENTS
-- =============================================================================

CREATE TABLE agents (
  id         SERIAL        PRIMARY KEY,
  cluster_id INTEGER       NOT NULL REFERENCES clusters (id) ON DELETE RESTRICT,
  name       VARCHAR(100)  NOT NULL,
  email      VARCHAR(150)  NOT NULL,
  role       VARCHAR(80)   NOT NULL DEFAULT 'CSR',
  risk_level risk_level    NOT NULL DEFAULT 'Safe',
  is_active  BOOLEAN       NOT NULL DEFAULT TRUE,
  created_by INTEGER       REFERENCES users (id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_agents_email UNIQUE (email)
);
CREATE TRIGGER trg_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- CALLS
-- =============================================================================

CREATE TABLE calls (
  id            SERIAL        PRIMARY KEY,
  uuid          UUID          NOT NULL DEFAULT gen_random_uuid(),
  agent_id      INTEGER       NOT NULL REFERENCES agents   (id) ON DELETE RESTRICT,
  cluster_id    INTEGER       NOT NULL REFERENCES clusters (id) ON DELETE RESTRICT,
  filename      VARCHAR(255)  NOT NULL,
  file_path     VARCHAR(512),
  file_size     BIGINT,
  duration_sec  SMALLINT,
  upload_status upload_status NOT NULL DEFAULT 'pending',
  uploaded_by   INTEGER       REFERENCES users (id) ON DELETE SET NULL,
  call_date     DATE,
  created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_calls_uuid UNIQUE (uuid)
);
CREATE INDEX ix_calls_agent         ON calls (agent_id);
CREATE INDEX ix_calls_cluster       ON calls (cluster_id);
CREATE INDEX ix_calls_call_date     ON calls (call_date);
CREATE INDEX ix_calls_upload_status ON calls (upload_status);
CREATE TRIGGER trg_calls_updated_at BEFORE UPDATE ON calls FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- ANALYSIS RESULTS
-- =============================================================================

CREATE TABLE analysis_results (
  id                     SERIAL        PRIMARY KEY,
  call_id                INTEGER       NOT NULL UNIQUE REFERENCES calls (id) ON DELETE CASCADE,
  predicted_emotion      emotion       NOT NULL,
  confidence             NUMERIC(5,4)  NOT NULL,
  all_probabilities      JSONB,
  valence                valence,
  arousal                arousal,
  risk_level             analysis_risk NOT NULL DEFAULT 'Low',
  transcription_text     TEXT,
  transcription_lang     VARCHAR(10)   DEFAULT 'en',
  transcription_duration NUMERIC(8,2),
  speaker_mode           VARCHAR(50),
  agent_channel          VARCHAR(10),
  caller_channel         VARCHAR(10),
  analyzed_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- CSR RECOMMENDATIONS
-- =============================================================================

CREATE TABLE csr_recommendations (
  id                 SERIAL        PRIMARY KEY,
  analysis_result_id INTEGER       NOT NULL UNIQUE REFERENCES analysis_results (id) ON DELETE CASCADE,
  action             csr_action    NOT NULL DEFAULT 'NONE',
  urgency            csr_urgency   NOT NULL DEFAULT 'LOW',
  reason             TEXT,
  instruction        TEXT,
  action_color       action_color,
  recommended_tone   TEXT,
  example_phrases    JSONB,
  do_list            JSONB,
  dont_list          JSONB,
  created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- ESCALATIONS
-- =============================================================================

CREATE TABLE escalations (
  id           SERIAL       PRIMARY KEY,
  call_id      INTEGER      NOT NULL REFERENCES calls  (id) ON DELETE CASCADE,
  agent_id     INTEGER      NOT NULL REFERENCES agents (id) ON DELETE CASCADE,
  escalated_to INTEGER      REFERENCES users (id) ON DELETE SET NULL,
  reason       TEXT,
  resolved     BOOLEAN      NOT NULL DEFAULT FALSE,
  resolved_at  TIMESTAMPTZ,
  resolved_by  INTEGER      REFERENCES users (id) ON DELETE SET NULL,
  notes        TEXT,
  created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_esc_call     ON escalations (call_id);
CREATE INDEX ix_esc_agent    ON escalations (agent_id);
CREATE INDEX ix_esc_resolved ON escalations (resolved);
CREATE TRIGGER trg_esc_updated_at BEFORE UPDATE ON escalations FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- AGENT DAILY STATS
-- =============================================================================

CREATE TABLE agent_daily_stats (
  id               SERIAL       PRIMARY KEY,
  agent_id         INTEGER      NOT NULL REFERENCES agents (id) ON DELETE CASCADE,
  stat_date        DATE         NOT NULL,
  calls_count      SMALLINT     NOT NULL DEFAULT 0,
  angry_count      SMALLINT     NOT NULL DEFAULT 0,
  frustrated_count SMALLINT     NOT NULL DEFAULT 0,
  neutral_count    SMALLINT     NOT NULL DEFAULT 0,
  happy_count      SMALLINT     NOT NULL DEFAULT 0,
  sad_count        SMALLINT     NOT NULL DEFAULT 0,
  escalations      SMALLINT     NOT NULL DEFAULT 0,
  avg_risk_score   NUMERIC(5,2) NOT NULL DEFAULT 0.00,
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_agent_daily UNIQUE (agent_id, stat_date)
);
CREATE INDEX ix_agent_daily_date ON agent_daily_stats (stat_date);
CREATE TRIGGER trg_agent_daily_updated_at BEFORE UPDATE ON agent_daily_stats FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- CLUSTER DAILY STATS
-- =============================================================================

CREATE TABLE cluster_daily_stats (
  id            SERIAL       PRIMARY KEY,
  cluster_id    INTEGER      NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
  stat_date     DATE         NOT NULL,
  calls_count   SMALLINT     NOT NULL DEFAULT 0,
  risky_agents  SMALLINT     NOT NULL DEFAULT 0,
  medium_agents SMALLINT     NOT NULL DEFAULT 0,
  safe_agents   SMALLINT     NOT NULL DEFAULT 0,
  escalations   SMALLINT     NOT NULL DEFAULT 0,
  created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_cluster_daily UNIQUE (cluster_id, stat_date)
);
CREATE INDEX ix_cluster_daily_date ON cluster_daily_stats (stat_date);
CREATE TRIGGER trg_cluster_daily_updated_at BEFORE UPDATE ON cluster_daily_stats FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- =============================================================================
-- USER SESSIONS
-- =============================================================================

CREATE TABLE user_sessions (
  id            SERIAL       PRIMARY KEY,
  user_id       INTEGER      NOT NULL REFERENCES users (id) ON DELETE CASCADE,
  refresh_token VARCHAR(512) NOT NULL,
  ip_address    VARCHAR(45),
  user_agent    VARCHAR(255),
  expires_at    TIMESTAMPTZ  NOT NULL,
  revoked       BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_sessions_user  ON user_sessions (user_id);
CREATE INDEX ix_sessions_token ON user_sessions (refresh_token);

-- =============================================================================
-- AUDIT LOGS
-- =============================================================================

CREATE TABLE audit_logs (
  id         BIGSERIAL    PRIMARY KEY,
  user_id    INTEGER      REFERENCES users (id) ON DELETE SET NULL,
  action     audit_action NOT NULL,
  table_name VARCHAR(64),
  record_id  INTEGER,
  old_values JSONB,
  new_values JSONB,
  ip_address VARCHAR(45),
  created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX ix_audit_user       ON audit_logs (user_id);
CREATE INDEX ix_audit_table      ON audit_logs (table_name, record_id);
CREATE INDEX ix_audit_created_at ON audit_logs (created_at);
