# micheline/intel/watchers.py
# Service de surveillance continue des sources (RSS/web/social)
# - Poll périodique basé sur le registry
# - Respect robots.txt (RFC 9309)
# - Rate limiting par domaine
# - Extraction + normalisation en raw events

import os
import sys
import time
import hashlib
import json
import sqlite3
import threading
import queue
from typing import Optional, List, Dict, Set
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import feedparser
import trafilatura
from bs4 import BeautifulSoup

import config
from micheline.intel.entity_registry import EntityRegistry

# Base de données des raw events (séparée du registry et du chat)
EVENTS_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "intel", "db", "raw_events.sqlite"
)


class RateLimiter:
    """Rate limiter par domaine (évite de marteler les serveurs)."""
    
    def __init__(self, min_interval_sec: float = 2.0):
        self.min_interval = min_interval_sec
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def wait_if_needed(self, domain: str):
        """Attend si nécessaire avant d'autoriser l'accès au domaine."""
        with self._lock:
            last = self._last_access.get(domain, 0.0)
            now = time.time()
            elapsed = now - last
            
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                time.sleep(wait_time)
            
            self._last_access[domain] = time.time()


class RobotsChecker:
    """Vérifie le respect de robots.txt (RFC 9309)."""
    
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or getattr(
            config, "WATCHER_USER_AGENT", 
            "MichelineBot/1.0 (Intelligence Gathering; +https://github.com/yourproject)"
        )
        self._parsers: Dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()
    
    def can_fetch(self, url: str) -> bool:
        """Vérifie si le bot peut fetch l'URL selon robots.txt."""
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            with self._lock:
                if base_url not in self._parsers:
                    rp = RobotFileParser()
                    rp.set_url(f"{base_url}/robots.txt")
                    try:
                        rp.read()
                        self._parsers[base_url] = rp
                    except Exception as e:
                        print(f"[Robots] Erreur lecture robots.txt pour {base_url}: {e}")
                        # En cas d'erreur, on autorise (principe de tolérance)
                        return True
                
                parser = self._parsers.get(base_url)
                if parser:
                    return parser.can_fetch(self.user_agent, url)
                
            return True
        except Exception as e:
            print(f"[Robots] Erreur vérification {url}: {e}")
            return True  # Tolérant en cas d'erreur


class RawEventsDB:
    """Base de données des événements bruts extraits (avant normalisation)."""
    
    def __init__(self, db_path: str = EVENTS_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
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
            
            # Table des événements bruts
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_events (
                event_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                source_id INTEGER,
                entity_id TEXT,
                source_url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                published_at TEXT,
                fetched_at TEXT NOT NULL,
                is_processed INTEGER DEFAULT 0,
                processing_status TEXT,
                metadata TEXT
            )
            """)
            
            # Index pour dédup et recherche
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_hash 
            ON raw_events(content_hash)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_entity 
            ON raw_events(entity_id)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_processed 
            ON raw_events(is_processed)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_fetched 
            ON raw_events(fetched_at DESC)
            """)
            
            conn.commit()
    
    def add_raw_event(
        self,
        source_id: int,
        entity_id: str,
        source_url: str,
        source_type: str,
        title: str,
        content: str,
        published_at: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Ajoute un événement brut (avec déduplication par hash de contenu).
        Retourne l'event_id si ajouté, None si déjà existant.
        """
        try:
            # Hash pour dédup (titre + contenu tronqué)
            dedup_text = (title or "") + (content or "")[:500]
            content_hash = hashlib.sha256(dedup_text.encode('utf-8')).hexdigest()
            
            event_id = f"{entity_id}_{int(time.time() * 1000)}_{content_hash[:8]}"
            
            with self._get_conn() as conn:
                conn.execute("""
                INSERT INTO raw_events (
                    event_id, content_hash, source_id, entity_id,
                    source_url, source_type, title, content,
                    published_at, fetched_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    content_hash,
                    source_id,
                    entity_id,
                    source_url,
                    source_type,
                    title,
                    content,
                    published_at or datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    json.dumps(metadata or {})
                ))
                conn.commit()
                return event_id
        except sqlite3.IntegrityError:
            # Déjà existant (hash collision)
            return None
        except Exception as e:
            print(f"[EventsDB] Erreur ajout événement: {e}")
            return None
    
    def get_unprocessed_events(self, limit: int = 100) -> List[Dict]:
        """Récupère les événements non encore traités (pour le bloc 3)."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                SELECT * FROM raw_events 
                WHERE is_processed = 0
                ORDER BY fetched_at DESC
                LIMIT ?
                """, (limit,))
                
                events = []
                for row in cursor.fetchall():
                    events.append({
                        "event_id": row["event_id"],
                        "content_hash": row["content_hash"],
                        "source_id": row["source_id"],
                        "entity_id": row["entity_id"],
                        "source_url": row["source_url"],
                        "source_type": row["source_type"],
                        "title": row["title"],
                        "content": row["content"],
                        "published_at": row["published_at"],
                        "fetched_at": row["fetched_at"],
                        "metadata": json.loads(row["metadata"] or "{}")
                    })
                
                return events
        except Exception as e:
            print(f"[EventsDB] Erreur récupération événements: {e}")
            return []
    
    def mark_as_processed(self, event_id: str, status: str = "ok"):
        """Marque un événement comme traité."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                UPDATE raw_events 
                SET is_processed = 1, processing_status = ?
                WHERE event_id = ?
                """, (status, event_id))
                conn.commit()
        except Exception as e:
            print(f"[EventsDB] Erreur marquage événement: {e}")
    
    def get_stats(self) -> Dict:
        """Statistiques de la base d'événements."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                total = cursor.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
                unprocessed = cursor.execute(
                    "SELECT COUNT(*) FROM raw_events WHERE is_processed = 0"
                ).fetchone()[0]
                
                # Dernière heure
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                last_hour = cursor.execute(
                    "SELECT COUNT(*) FROM raw_events WHERE fetched_at >= ?",
                    (one_hour_ago,)
                ).fetchone()[0]
                
                return {
                    "total_events": total,
                    "unprocessed": unprocessed,
                    "last_hour": last_hour
                }
        except Exception as e:
            print(f"[EventsDB] Erreur stats: {e}")
            return {"total_events": 0, "unprocessed": 0, "last_hour": 0}


class SourceWatcher:
    """Watcher individuel pour une source (RSS/web)."""
    
    def __init__(
        self,
        source_data: Dict,
        events_db: RawEventsDB,
        rate_limiter: RateLimiter,
        robots_checker: RobotsChecker
    ):
        self.source_data = source_data
        self.events_db = events_db
        self.rate_limiter = rate_limiter
        self.robots_checker = robots_checker
        
        self.source_id = source_data["source_id"]
        self.entity_id = source_data["entity_id"]
        self.entity_name = source_data["entity_name"]
        self.source_type = source_data["source_type"]
        self.url = source_data["url"]
        self.trust_score = source_data["trust_score"]
        
        self._last_check: Optional[datetime] = None
    
    def should_check(self, interval_minutes: int) -> bool:
        """Détermine si on doit vérifier cette source maintenant."""
        if self._last_check is None:
            return True
        
        elapsed = datetime.now() - self._last_check
        return elapsed.total_seconds() >= (interval_minutes * 60)
    
    def fetch(self) -> List[Dict]:
        """
        Fetch la source et retourne une liste d'événements bruts.
        Gère RSS et pages web classiques.
        """
        domain = urlparse(self.url).netloc
        
        # Rate limiting
        self.rate_limiter.wait_if_needed(domain)
        
        # Vérification robots.txt
        if not self.robots_checker.can_fetch(self.url):
            print(f"[Watcher] ⛔ Bloqué par robots.txt: {self.url}")
            return []
        
        self._last_check = datetime.now()
        
        if self.source_type == "rss":
            return self._fetch_rss()
        elif self.source_type in ("website", "official_doc"):
            return self._fetch_web()
        elif self.source_type == "social":
            return self._fetch_social()
        else:
            print(f"[Watcher] Type non supporté: {self.source_type}")
            return []
    
    def _fetch_rss(self) -> List[Dict]:
        """Fetch un flux RSS."""
        try:
            headers = {
                'User-Agent': getattr(
                    config, "WATCHER_USER_AGENT",
                    "MichelineBot/1.0"
                )
            }
            
            print(f"[Watcher] 📡 RSS: {self.entity_name} — {self.url}")
            
            feed = feedparser.parse(self.url, request_headers=headers)
            
            if feed.bozo:  # Erreur de parsing
                print(f"[Watcher] ⚠ Erreur parsing RSS: {feed.bozo_exception}")
                return []
            
            events = []
            for entry in feed.entries[:20]:  # Limite aux 20 derniers items
                title = entry.get('title', '')
                link = entry.get('link', self.url)
                
                # Contenu (résumé ou description)
                content = entry.get('summary', '') or entry.get('description', '')
                
                # Date de publication
                published = entry.get('published_parsed') or entry.get('updated_parsed')
                published_dt = None
                if published:
                    try:
                        published_dt = datetime(*published[:6]).isoformat()
                    except Exception:
                        pass
                
                if not content or len(content.strip()) < 50:
                    # Pas assez de contenu, on skip
                    continue
                
                events.append({
                    "title": title,
                    "content": content,
                    "source_url": link,
                    "published_at": published_dt,
                    "metadata": {
                        "feed_title": feed.feed.get('title', ''),
                        "entry_id": entry.get('id', ''),
                    }
                })
            
            print(f"[Watcher] ✅ {len(events)} items extraits du RSS")
            return events
            
        except Exception as e:
            print(f"[Watcher] ❌ Erreur fetch RSS {self.url}: {e}")
            return []
    
    def _fetch_web(self) -> List[Dict]:
        """Fetch une page web classique et extrait le texte principal."""
        try:
            headers = {
                'User-Agent': getattr(
                    config, "WATCHER_USER_AGENT",
                    "MichelineBot/1.0"
                )
            }
            
            print(f"[Watcher] 🌐 WEB: {self.entity_name} — {self.url}")
            
            r = requests.get(
                self.url,
                headers=headers,
                timeout=15,
                allow_redirects=True
            )
            r.raise_for_status()
            
            # Extraction texte principal (trafilatura)
            content = trafilatura.extract(
                r.text,
                include_comments=False,
                include_tables=True
            )
            
            # Fallback BeautifulSoup si trafilatura échoue
            if not content or len(content.strip()) < 100:
                soup = BeautifulSoup(r.text, 'lxml')
                
                # Retire scripts/styles
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                
                content = soup.get_text(separator='\n', strip=True)
            
            if not content or len(content.strip()) < 100:
                print(f"[Watcher] ⚠ Contenu insuffisant extrait de {self.url}")
                return []
            
            # Titre (via BeautifulSoup)
            soup = BeautifulSoup(r.text, 'lxml')
            title = ""
            if soup.title:
                title = soup.title.string or ""
            
            events = [{
                "title": title.strip(),
                "content": content.strip(),
                "source_url": self.url,
                "published_at": None,  # Pas de date précise sur page web
                "metadata": {
                    "content_length": len(content),
                }
            }]
            
            print(f"[Watcher] ✅ Contenu extrait ({len(content)} chars)")
            return events
            
        except Exception as e:
            print(f"[Watcher] ❌ Erreur fetch web {self.url}: {e}")
            return []
    
    def _fetch_social(self) -> List[Dict]:
        """
        Fetch réseau social (Truth Social via RSS public).
        Note: pour un vrai scraping, utiliser un client API ou RSS tiers.
        """
        # Pour Truth Social, on peut utiliser un flux RSS public de suivi
        # Exemple: rsshub.app/truthsocial/realDonaldTrump
        
        # Si l'URL est déjà un RSS, on délègue
        if "rss" in self.url.lower() or "feed" in self.url.lower():
            return self._fetch_rss()
        
        # Sinon, on tente une extraction web classique
        # (À affiner selon la plateforme)
        print(f"[Watcher] ℹ️ Social ({self.source_type}): utiliser un flux RSS dédié recommandé")
        return []


class WatcherService:
    """
    Service principal de surveillance continue.
    - Lit le registry pour savoir quoi surveiller
    - Lance des watchers pour chaque source active
    - Écrit les raw events dans la base
    """
    
    def __init__(self):
        self.registry = EntityRegistry()
        self.events_db = RawEventsDB()
        self.rate_limiter = RateLimiter(
            min_interval_sec=float(getattr(config, "WATCHER_RATE_LIMIT_SEC", 2.0))
        )
        self.robots_checker = RobotsChecker()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Intervalles de polling selon le type de source (minutes)
        self.poll_intervals = {
            "rss": int(getattr(config, "WATCHER_RSS_INTERVAL_MIN", 5)),
            "website": int(getattr(config, "WATCHER_WEB_INTERVAL_MIN", 15)),
            "official_doc": int(getattr(config, "WATCHER_OFFICIAL_INTERVAL_MIN", 30)),
            "social": int(getattr(config, "WATCHER_SOCIAL_INTERVAL_MIN", 3)),
        }
    
    def start(self, daemon: bool = True):
        """Démarre le service en arrière-plan."""
        if self._running:
            print("[WatcherService] Déjà en cours d'exécution.")
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._run_loop, daemon=daemon)
        self._thread.start()
        
        print("[WatcherService] ✅ Service de surveillance démarré.")
    
    def stop(self):
        """Arrête le service proprement."""
        if not self._running:
            return
        
        print("[WatcherService] Arrêt en cours...")
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        print("[WatcherService] ✅ Service arrêté.")
    
    def _run_loop(self):
        """Boucle principale de surveillance."""
        print("[WatcherService] Boucle de surveillance active.")
        
        # Cache des watchers (pour mémoriser last_check)
        watchers_cache: Dict[int, SourceWatcher] = {}
        
        while self._running and not self._stop_event.is_set():
            try:
                # Récupère toutes les sources actives du registry
                sources = self.registry.list_all_active_sources()
                
                if not sources:
                    print("[WatcherService] Aucune source active dans le registry.")
                    time.sleep(60)  # Attend 1 minute avant de réessayer
                    continue
                
                print(f"\n[WatcherService] 🔍 Scan de {len(sources)} sources actives...")
                
                new_events_count = 0
                
                for source in sources:
                    if not self._running:
                        break
                    
                    source_id = source["source_id"]
                    
                    # Crée ou récupère le watcher
                    if source_id not in watchers_cache:
                        watchers_cache[source_id] = SourceWatcher(
                            source_data=source,
                            events_db=self.events_db,
                            rate_limiter=self.rate_limiter,
                            robots_checker=self.robots_checker
                        )
                    
                    watcher = watchers_cache[source_id]
                    
                    # Détermine l'intervalle selon le type
                    interval = self.poll_intervals.get(
                        source["source_type"],
                        15  # Défaut: 15 minutes
                    )
                    
                    # Vérifie si on doit poller cette source maintenant
                    if not watcher.should_check(interval):
                        continue
                    
                    # Fetch la source
                    try:
                        raw_events = watcher.fetch()
                        
                        # Ajoute les événements à la base
                        for event in raw_events:
                            event_id = self.events_db.add_raw_event(
                                source_id=source_id,
                                entity_id=source["entity_id"],
                                source_url=event["source_url"],
                                source_type=source["source_type"],
                                title=event.get("title", ""),
                                content=event["content"],
                                published_at=event.get("published_at"),
                                metadata=event.get("metadata", {})
                            )
                            
                            if event_id:
                                new_events_count += 1
                    
                    except Exception as e:
                        print(f"[WatcherService] ⚠ Erreur watcher {source['entity_name']}: {e}")
                        continue
                
                # Stats
                if new_events_count > 0:
                    print(f"\n[WatcherService] 📊 {new_events_count} nouveaux événements capturés.")
                
                stats = self.events_db.get_stats()
                print(f"[WatcherService] Total events: {stats['total_events']} | "
                      f"Non traités: {stats['unprocessed']} | "
                      f"Dernière heure: {stats['last_hour']}")
                
                # Pause avant le prochain cycle (cycle global: 1 minute)
                # Les sources individuelles ont leurs propres intervalles
                time.sleep(60)
            
            except Exception as e:
                print(f"[WatcherService] ❌ Erreur dans la boucle principale: {e}")
                time.sleep(60)
        
        print("[WatcherService] Boucle terminée.")
    
    def get_status(self) -> Dict:
        """Retourne le statut du service."""
        stats = self.events_db.get_stats()
        
        return {
            "running": self._running,
            "total_events": stats["total_events"],
            "unprocessed_events": stats["unprocessed"],
            "events_last_hour": stats["last_hour"],
            "active_sources": len(self.registry.list_all_active_sources())
        }


# ========== Point d'entrée CLI (optionnel) ==========

if __name__ == "__main__":
    import signal
    
    service = WatcherService()
    
    # Gestion propre du Ctrl+C
    def signal_handler(sig, frame):
        print("\n[WatcherService] Interruption détectée (Ctrl+C)...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Micheline Watcher Service ===")
    print("Démarrage du service de surveillance...")
    print("Appuyez sur Ctrl+C pour arrêter.\n")
    
    service.start(daemon=False)  # Mode non-daemon pour CLI
    
    # Boucle pour garder le programme actif
    try:
        while service._running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    service.stop()