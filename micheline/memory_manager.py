# micheline/memory_manager.py
# Gère la mémoire persistante de l'assistant (conversations, profil) via une base SQLite.

import sqlite3
import os
from typing import List, Dict, Optional
from datetime import datetime

def get_current_datetime_str() -> str:
    """
    Retourne la date et l'heure actuelles du PC
    sous un format lisible (ex: 'Mardi 4 Février 2025, 14:37:12').
    """
    # On utilise strftime pour la mise en forme
    now = datetime.now()
    return now.strftime("%A %d %B %Y, %H:%M:%S")


# Chemin ABSOLU (évite les soucis de répertoire courant)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE_DIR, "memory", "db", "assistant.sqlite")

class MemoryManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        # Timeout plus long pour éviter "database is locked" en multi-threads
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        """Crée les tables si elles n'existent pas."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # Table pour l'historique des conversations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
            """)
            # Table simple clé-valeur pour le profil utilisateur
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """)
            conn.commit()

    def add_message(self, role: str, content: str):
        """Ajoute un message (user ou assistant) à l'historique avec horodatage local fiable."""
        ts = get_current_datetime_str()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversation_history (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, ts)
            )
            conn.commit()

    def get_last_messages(self, limit: int = 10) -> List[Dict[str, str]]:
        """Récupère les N derniers messages pour les injecter dans le contexte du LLM (avec timestamp)."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT role, content, timestamp FROM conversation_history ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            messages = [
                {"role": row[0], "content": row[1], "timestamp": row[2]}
                for row in cursor.fetchall()
            ]
            # On renvoie dans l'ordre chronologique (du plus ancien au plus récent)
            return list(reversed(messages))

    def set_profile_value(self, key: str, value: str):
        """Sauvegarde une préférence dans le profil utilisateur."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)",
                (key, value)
            )
            conn.commit()

    def get_profile_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Récupère une préférence du profil utilisateur."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT value FROM user_profile WHERE key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else default

    def clear_history(self):
        """Efface l'historique de conversation."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM conversation_history")
            conn.commit()