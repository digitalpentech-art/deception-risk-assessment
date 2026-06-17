CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    risk_score FLOAT
);

CREATE TABLE IF NOT EXISTS emotions (
    session_id INTEGER,
    fear FLOAT,
    happy FLOAT,
    sad FLOAT,
    anger FLOAT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS voice (
    session_id INTEGER,
    pitch FLOAT,
    energy FLOAT,
    stress_level TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
