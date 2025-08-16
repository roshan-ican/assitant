# data/storage.py - SQLite for ML data only
import sqlite3
from datetime import datetime

class TaskStorage:
    def __init__(self, db_path="tasks.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table for ML training data only"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_tasks (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT,
                    text TEXT,
                    created_at TIMESTAMP,
                    notion_page_id TEXT
                )
            """)

    def save_ml_data(self, user_id, task_text, notion_page_id):
        """Save task data for ML training"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO ml_tasks (user_id, text, created_at, notion_page_id) VALUES (?, ?, ?, ?)",
                (user_id, task_text, datetime.now(), notion_page_id)
            )
            return cursor.lastrowid

    def get_user_tasks_for_ml(self, user_id, limit=100):
        """Get user tasks for ML training"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT text, created_at FROM ml_tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
            return cursor.fetchall()