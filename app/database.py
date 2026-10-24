
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "health_twin.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                height_cm REAL NOT NULL,
                baseline_weight_kg REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                weight_kg REAL,
                resting_hr_bpm INTEGER,
                glucose_mg_dl REAL,
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id)
                    ON DELETE CASCADE
            )
        """)
