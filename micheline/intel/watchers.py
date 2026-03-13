# micheline/intel/watchers.py
# Service de surveillance continue des sources (RSS/web/social/official_doc)
# - Poll périodique basé sur le registry (EntityRegistry SQLite)
# - Respect robots.txt (tolérant en cas d'erreur)
# - Rate limiting par domaine
# - Extraction + stockage en "raw events" (SQLite)
# - Callback optionnel (on_item) pour pousser les lectures vers l'UI (onglet News)
# - Rétention (ex: 7 jours) + purge automatique
# - Replay au démarrage avec timestamp ORIGINAL (fetched_at) => l'heure affichée reste correcte après redémarrage

from __future__ import annotations

import os
import time
import json
import uuid
import hashlib
import sqlite3
import threading
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import feedparser

try:
    import certifi
except Exception:
    certifi = None

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

import config
from micheline.intel.entity_registry import EntityRegistry


# ==========================
# Constantes / chemins
# ==========================

EVENTS_DB_PATH = os.path.join(os.path.dirname(__file__), "db", "raw_events.sqlite")

DEFAULT_UA = getattr(
    config,
    "WATCHER_USER_AGENT",
    "MichelineBot/1.0 (Intelligence Gathering)",
)

# Env: 0 => désactive verify SSL (pas recommandé)
SSL_VERIFY = os.getenv("MICHELINE_SSL_VERIFY", "1").strip() != "0"


# ==========================
# Utils
# ==========================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

def _requests_verify_value() -> Any:
    """
    verify= pour requests:
    - False si MICHELINE_SSL_VERIFY=0
    - sinon certifi.where() si dispo
    - sinon True
    """
    if not SSL_VERIFY:
        return False
    if certifi is not None:
        try:
            return certifi.where()
        except Exception:
            pass
    return True


# ==========================
# URL Rewrites (sources "propres" vs robots.txt)
# ==========================

TRUMP_TRUTH_RSS = "https://trumpstruth.org/feed"

# RSS Google News (évite de scraper opec.org directement si robots bloque)
OPEC_GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?"
    "q=site%3Aopec.org%20press%20releases&hl=en-US&gl=US&ceid=US:en"
)

def _rewrite_source_if_needed(source_type: str, url: str) -> (str, str, str):
    st = (source_type or "").lower().strip()
    u = (url or "").strip()
    ul = u.lower()

    # Trump: TruthSocial / RSSHub -> trumpstruth RSS (public)
    if "truthsocial.com/@realdonaldtrump" in ul:
        return ("rss", TRUMP_TRUTH_RSS, "TruthSocial bloqué robots -> trumpstruth RSS")
    if "rsshub.app/truthsocial/realdonaldtrump" in ul:
        return ("rss", TRUMP_TRUTH_RSS, "RSSHub bloqué robots -> trumpstruth RSS")

    # OPEC press_room -> Google News RSS (site:opec.org)
    if "opec.org/opec_web/en/press_room/" in ul:
        return ("rss", OPEC_GOOGLE_NEWS_RSS, "OPEC press_room bloqué robots -> Google News RSS")

    return (st or source_type, u, "")


# ==========================
# Rate Limiter
# ==========================

class RateLimiter:
    def __init__(self, min_interval_sec: float = 2.0):
        self.min_interval = float(min_interval_sec or 0.0)
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait_if_needed(self, domain: str):
        if not self.min_interval or not domain:
            return
        with self._lock:
            last = self._last_access.get(domain, 0.0)
            now = time.time()
            elapsed = now - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_access[domain] = time.time()


# ==========================
# Robots checker
# ==========================

class RobotsChecker:
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or DEFAULT_UA
        self._parsers: Dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()

    def can_fetch(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            if not parsed.scheme or not parsed.netloc:
                return True

            with self._lock:
                rp = self._parsers.get(base_url)
                if rp is None:
                    rp = RobotFileParser()
                    rp.set_url(f"{base_url}/robots.txt")
                    try:
                        rp.read()
                    except Exception as e:
                        print(f"[Robots] Erreur lecture robots.txt pour {base_url}: {e}")
                        return True
                    self._parsers[base_url] = rp

            parser = self._parsers.get(base_url)
            if parser:
                return parser.can_fetch(self.user_agent, url)
            return True
        except Exception as e:
            print(f"[Robots] Erreur vérification {url}: {e}")
            return True


# ==========================
# RawEventsDB (persist + rétention + replay)
# ==========================

class RawEventsDB:
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

    def _ensure_column(self, conn, table: str, col: str, coltype: str):
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                conn.commit()
                print(f"[RawEventsDB] Migration: ajout colonne {table}.{col}")
        except Exception as e:
            print(f"[RawEventsDB] Migration colonne échouée ({table}.{col}): {e}")

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute(
                """
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

                    url TEXT,
                    metadata TEXT,

                    is_processed INTEGER DEFAULT 0,
                    processing_status TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_fetched_at ON raw_events(fetched_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_entity_id ON raw_events(entity_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_processed ON raw_events(is_processed)")
            conn.commit()

            # Migration auto (si DB ancienne)
            self._ensure_column(conn, "raw_events", "url", "TEXT")
            self._ensure_column(conn, "raw_events", "metadata", "TEXT")
            self._ensure_column(conn, "raw_events", "published_at", "TEXT")
            self._ensure_column(conn, "raw_events", "processing_status", "TEXT")

    def insert_if_new(self, event: Dict[str, Any]) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO raw_events (
                        event_id, content_hash,
                        source_id, entity_id,
                        source_url, source_type,
                        title, content,
                        published_at, fetched_at,
                        url, metadata,
                        is_processed, processing_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.get("event_id"),
                        event.get("content_hash"),
                        event.get("source_id"),
                        event.get("entity_id"),
                        event.get("source_url"),
                        event.get("source_type"),
                        event.get("title"),
                        event.get("content"),
                        event.get("published_at"),
                        event.get("fetched_at"),
                        event.get("url"),
                        event.get("metadata"),
                        int(event.get("is_processed", 0)),
                        event.get("processing_status"),
                    ),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"[RawEventsDB] Erreur insertion: {e}")
            return False

    def purge_older_than_days(self, days: int = 7) -> int:
        """Supprime les raw_events dont fetched_at est plus vieux que N jours."""
        try:
            days = int(days)
            if days <= 0:
                return 0
        except Exception:
            days = 7

        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM raw_events WHERE fetched_at < datetime('now', ?)",
                    (f"-{days} days",),
                )
                deleted = cur.rowcount if cur.rowcount is not None else 0
                conn.commit()
                if deleted:
                    print(f"[RawEventsDB] Purge: {deleted} event(s) supprimé(s) (> {days} jours)")
                return int(deleted)
        except Exception as e:
            print(f"[RawEventsDB] Purge erreur: {e}")
            return 0

    def list_recent_for_ui(self, days: int = 7, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Retourne des items récents pour re-remplir l'onglet News au démarrage.
        IMPORTANT: on renvoie 'read_at' = fetched_at (original).
        """
        try:
            days = int(days)
        except Exception:
            days = 7
        try:
            limit = int(limit)
        except Exception:
            limit = 500

        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT fetched_at, title, url, source_url, metadata
                    FROM raw_events
                    WHERE fetched_at >= datetime('now', ?)
                    ORDER BY fetched_at DESC
                    LIMIT ?
                    """,
                    (f"-{days} days", limit),
                ).fetchall()

            out = []
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r["metadata"] or "{}")
                except Exception:
                    meta = {}

                url = (r["url"] or r["source_url"] or "").strip()
                site = (meta.get("domain") or _domain(url) or _domain(r["source_url"] or "") or "unknown").strip()
                title = (r["title"] or "").strip() or "(sans titre)"

                out.append(
                    {
                        "read_at": r["fetched_at"],  # <- timestamp original
                        "site": site,
                        "title": title,
                        "url": url,
                    }
                )
            return out
        except Exception as e:
            print(f"[RawEventsDB] Erreur list_recent_for_ui: {e}")
            return []


# ==========================
# Watcher (1 source)
# ==========================

class Watcher:
    def __init__(
        self,
        source: Dict[str, Any],
        rate_limiter: RateLimiter,
        robots_checker: RobotsChecker,
        user_agent: str = None,
        timeout_sec: int = 20,
    ):
        self.source = source
        self.source_id = source.get("source_id")
        self.entity_id = source.get("entity_id")
        self.entity_name = source.get("entity_name", "")
        self.source_type = (source.get("source_type") or "rss").lower()
        self.url = source.get("url", "")
        self.user_agent = user_agent or DEFAULT_UA
        self.timeout_sec = int(timeout_sec or 20)

        self.rate_limiter = rate_limiter
        self.robots_checker = robots_checker

    def fetch(self) -> List[Dict[str, Any]]:
        if not self.url:
            return []

        # Rewrite automatique (robots / sources cassées)
        new_type, new_url, reason = _rewrite_source_if_needed(self.source_type, self.url)
        if reason and new_url and (new_url != self.url or new_type != self.source_type):
            print(f"[Watcher] Rewrite source: {self.url} -> {new_url} ({reason})")
            self.url = new_url
            self.source_type = new_type

        if not self.robots_checker.can_fetch(self.url):
            print(f"[Watcher] robots.txt interdit: {self.url}")
            return []

        st = self.source_type
        if st == "rss":
            return self._fetch_rss()
        if st in ("website", "official_doc"):
            return self._fetch_webpage()
        if st == "social":
            return self._fetch_social()

        return self._fetch_rss()

    def _http_get(self, url: str) -> requests.Response:
        headers = {"User-Agent": self.user_agent}
        self.rate_limiter.wait_if_needed(_domain(url))
        return requests.get(
            url,
            headers=headers,
            timeout=self.timeout_sec,
            verify=_requests_verify_value(),
        )

    def _extract_text_from_html(self, html: str, url: str = "") -> str:
        if trafilatura is not None:
            try:
                txt = trafilatura.extract(html, include_comments=False, include_tables=False)
                if txt and txt.strip():
                    return txt.strip()
            except Exception:
                pass

        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                txt = soup.get_text("\n")
                txt = "\n".join(line.strip() for line in txt.splitlines() if line.strip())
                return txt.strip()
            except Exception:
                pass

        return (html or "").strip()

    def _make_event(
        self,
        title: str,
        content: str,
        published_at: Optional[str],
        item_url: Optional[str],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fetched_at = _now_str()
        item_url = (item_url or "").strip()
        content = (content or "").strip()
        title = (title or "").strip()

        basis = f"{self.entity_id}|{self.source_id}|{self.source_type}|{item_url}|{title}|{content[:2000]}"
        content_hash = _sha256(basis)

        meta = {"entity_name": self.entity_name, "domain": _domain(item_url or self.url)}
        if extra_meta:
            meta.update(extra_meta)

        return {
            "event_id": str(uuid.uuid4()),
            "content_hash": content_hash,
            "source_id": self.source_id,
            "entity_id": self.entity_id,
            "source_url": self.url,
            "source_type": self.source_type,
            "title": title,
            "content": content,
            "published_at": published_at,
            "fetched_at": fetched_at,
            "url": item_url or self.url,
            "metadata": _safe_json(meta),
            "is_processed": 0,
            "processing_status": None,
        }

    def _fetch_rss(self) -> List[Dict[str, Any]]:
        try:
            r = self._http_get(self.url)
            r.raise_for_status()

            feed = feedparser.parse(r.content)
            entries = getattr(feed, "entries", []) or []
            out: List[Dict[str, Any]] = []

            fetch_full = bool(getattr(config, "WATCHER_RSS_FETCH_FULL_ARTICLE", False))
            max_items = int(getattr(config, "WATCHER_RSS_MAX_ITEMS", 30))

            for e in entries[:max_items]:
                title = (getattr(e, "title", "") or "").strip()
                link = (getattr(e, "link", "") or "").strip()
                published = (getattr(e, "published", "") or "").strip() or (getattr(e, "updated", "") or "").strip()

                summary = (getattr(e, "summary", "") or "").strip()
                content = summary

                try:
                    if hasattr(e, "content") and e.content:
                        content_val = e.content[0].value
                        if isinstance(content_val, str) and content_val.strip():
                            content = content_val
                except Exception:
                    pass

                if fetch_full and link and self.robots_checker.can_fetch(link):
                    try:
                        rr = self._http_get(link)
                        if rr.status_code < 400:
                            extracted = self._extract_text_from_html(rr.text, url=link)
                            if extracted and len(extracted) > 200:
                                content = extracted
                    except Exception:
                        pass

                if not content:
                    continue

                out.append(
                    self._make_event(
                        title=title,
                        content=content,
                        published_at=published or None,
                        item_url=link or self.url,
                        extra_meta={"rss_url": self.url},
                    )
                )

            return out
        except Exception as e:
            print(f"[WatcherService] Source error ({self.url}): {e}")
            return []

    def _fetch_webpage(self) -> List[Dict[str, Any]]:
        try:
            r = self._http_get(self.url)
            r.raise_for_status()

            text = self._extract_text_from_html(r.text, url=self.url)
            if not text:
                return []

            title = ""
            if BeautifulSoup is not None:
                try:
                    soup = BeautifulSoup(r.text, "html.parser")
                    if soup.title and soup.title.text:
                        title = soup.title.text.strip()
                except Exception:
                    pass

            return [
                self._make_event(
                    title=title or f"Website: {self.url}",
                    content=text,
                    published_at=None,
                    item_url=self.url,
                    extra_meta={"content_length": len(text)},
                )
            ]
        except Exception as e:
            print(f"[Watcher] ❌ Erreur fetch web {self.url}: {e}")
            return []

    def _fetch_social(self) -> List[Dict[str, Any]]:
        u = (self.url or "").lower()
        if "rss" in u or "feed" in u:
            return self._fetch_rss()
        print("[Watcher] ℹ️ Social: utiliser un flux RSS dédié recommandé")
        return []


# ==========================
# WatcherService
# ==========================

class WatcherService:
    """
    - Poll les sources actives dans EntityRegistry (SQLite)
    - Stocke dans raw_events.sqlite
    - Purge > WATCHER_RETENTION_DAYS (par défaut 7)
    - Replay au démarrage avec timestamp ORIGINAL (fetched_at) => UI cohérente
    """

    def __init__(self, on_item: Optional[Callable[..., None]] = None):
        self.registry = EntityRegistry()
        self.events_db = RawEventsDB()

        self.rate_limiter = RateLimiter(min_interval_sec=float(getattr(config, "WATCHER_RATE_LIMIT_SEC", 2.0)))
        self.robots_checker = RobotsChecker()

        self.on_item = on_item
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.poll_intervals = {
            "rss": int(getattr(config, "WATCHER_RSS_INTERVAL_MIN", 5)),
            "website": int(getattr(config, "WATCHER_WEB_INTERVAL_MIN", 15)),
            "official_doc": int(getattr(config, "WATCHER_OFFICIAL_INTERVAL_MIN", 30)),
            "social": int(getattr(config, "WATCHER_SOCIAL_INTERVAL_MIN", 3)),
        }

        self._last_poll: Dict[int, float] = {}

        # Rétention / purge
        self.retention_days = int(getattr(config, "WATCHER_RETENTION_DAYS", 7))
        self.purge_every_sec = int(getattr(config, "WATCHER_PURGE_EVERY_SEC", 3600))
        self._last_purge_ts = 0.0

        # Replay UI
        self._replayed_recent = False
        self.replay_limit = int(getattr(config, "WATCHER_REPLAY_LIMIT", 500))

    def start(self, daemon: bool = True):
        if self._running:
            print("[WatcherService] Déjà en cours d'exécution.")
            return

        self._running = True
        self._stop_event.clear()

        # Replay des news persistées (pour que l'onglet "News" survive aux redémarrages)
        self._replay_recent_news_once()

        self._thread = threading.Thread(target=self._run_loop, daemon=daemon)
        self._thread.start()
        print("[WatcherService] ✅ Service de surveillance démarré.")

    def stop(self):
        self._stop_event.set()
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _run_loop(self):
        print("[WatcherService] Boucle de surveillance active.")
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:
                print(f"[WatcherService] Erreur boucle: {e}")
            self._stop_event.wait(2.0)

    def _maybe_purge(self):
        try:
            now_ts = time.time()
            if (now_ts - self._last_purge_ts) >= max(60, self.purge_every_sec):
                self.events_db.purge_older_than_days(self.retention_days)
                self._last_purge_ts = now_ts
        except Exception:
            pass

    def _replay_recent_news_once(self):
        if self._replayed_recent:
            return
        self._replayed_recent = True

        if not callable(self.on_item):
            return

        try:
            items = self.events_db.list_recent_for_ui(days=self.retention_days, limit=self.replay_limit)
            # items est en DESC, on l'envoie en chrono (vieux -> récent) pour l'UI
            items = list(reversed(items))

            sent = 0
            for it in items:
                # IMPORTANT: on envoie fetched_at ORIGINAL via un "event" minimal
                ev = {"fetched_at": it.get("read_at")}
                try:
                    self.on_item(it["site"], it["title"], it["url"], ev)
                except TypeError:
                    # compat ancien callback (3 args)
                    try:
                        self.on_item(it["site"], it["title"], it["url"])
                    except Exception:
                        pass
                except Exception:
                    pass
                sent += 1

            if sent:
                print(f"[WatcherService] Replay UI: {sent} news (<= {self.retention_days} jours)")
        except Exception as e:
            print(f"[WatcherService] Replay UI error: {e}")

    def poll_once(self):
        # Purge périodique (rétention)
        self._maybe_purge()

        sources = self.registry.list_all_active_sources()
        if not sources:
            print("[WatcherService] Aucune source active dans le registry.")
            self._stop_event.wait(60.0)
            return

        now = time.time()

        for src in sources:
            if self._stop_event.is_set():
                return

            st = (src.get("source_type") or "rss").lower()
            interval_min = int(self.poll_intervals.get(st, 10))
            interval_sec = max(15, interval_min * 60)

            sid = src.get("source_id")
            if sid is None:
                due = True
            else:
                last = self._last_poll.get(int(sid), 0.0)
                due = (now - last) >= interval_sec

            if not due:
                continue

            try:
                watcher = Watcher(
                    source=src,
                    rate_limiter=self.rate_limiter,
                    robots_checker=self.robots_checker,
                    user_agent=DEFAULT_UA,
                    timeout_sec=int(getattr(config, "WATCHER_TIMEOUT_SEC", 20)),
                )
                events = watcher.fetch()
            except Exception as e:
                print(f"[WatcherService] Erreur watcher pour {src.get('url')}: {e}")
                events = []

            if sid is not None:
                self._last_poll[int(sid)] = time.time()

            for ev in events:
                if self._stop_event.is_set():
                    return

                inserted = self.events_db.insert_if_new(ev)
                if not inserted:
                    continue

                t = (ev.get("title") or "").strip()
                u = (ev.get("url") or ev.get("source_url") or "").strip()
                print(f"[Watcher] ✅ New raw event: {t[:120]} | {u}")

                self._emit_ui(ev)

    def _emit_ui(self, ev: Dict[str, Any]):
        if not callable(self.on_item):
            return
        try:
            url = (ev.get("url") or ev.get("source_url") or "").strip()
            site = _domain(url) or _domain(ev.get("source_url", "")) or "unknown"
            title = (ev.get("title") or "").strip() or "(sans titre)"

            # Supporte on_item(site,title,url,event) ou on_item(site,title,url)
            try:
                self.on_item(site, title, url, ev)
                return
            except TypeError:
                pass

            try:
                self.on_item(site, title, url)
                return
            except TypeError:
                pass

            self.on_item({"site": site, "title": title, "url": url, "event": ev})
        except Exception:
            pass


__all__ = ["RateLimiter", "RobotsChecker", "RawEventsDB", "Watcher", "WatcherService"]
