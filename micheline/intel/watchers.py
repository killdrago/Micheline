import os
import json
import time
import threading
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
import feedparser


# -------------------------
# Helpers
# -------------------------

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src"
}

def canonicalize_url(url: str) -> str:
    """Supprime les paramètres de tracking, normalise un peu l’URL."""
    try:
        url = (url or "").strip()
        if not url:
            return ""
        p = urlparse(url)
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k not in _TRACKING_PARAMS]
        new_query = urlencode(q, doseq=True)
        # Normalisation simple : retire fragment, garde scheme/netloc/path/query
        return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, ""))
    except Exception:
        return (url or "").strip()

def get_site_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# -------------------------
# Registry model
# -------------------------

@dataclass
class WatchSource:
    url: str
    type: str = "rss"             # "rss" (pour l’instant)
    active: bool = True
    label: str = ""               # nom sympa affiché si tu veux
    entity_id: str = ""           # optionnel
    poll_interval_sec: int = 0    # 0 => utilise l’intervalle global


# -------------------------
# Watcher Service
# -------------------------

class WatcherService:
    """
    Watcher always-on (RSS) :
    - lit un registry JSON
    - poll les RSS
    - dédup en SQLite
    - appelle on_read(site, title, url) quand il découvre un nouvel article
    """

    def __init__(
        self,
        registry_path: str = "micheline/intel/entities.json",
        db_path: str = "micheline/intel/db/news_reads.sqlite",
        poll_interval_sec: int = 120,
        on_read: Optional[Callable[[str, str, str], None]] = None,
        user_agent: str = "MichelineWatcher/1.0",
        timeout_sec: int = 20,
    ):
        self.registry_path = registry_path
        self.db_path = db_path
        self.poll_interval_sec = int(poll_interval_sec or 120)
        self.on_read = on_read
        self.user_agent = user_agent
        self.timeout_sec = int(timeout_sec or 20)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # évite de spammer "aucune source active" à chaque tick
        self._last_no_sources_log = 0.0

        self._ensure_db()

    # ------------- lifecycle -------------

    def start(self) -> None:
        """Démarre la boucle dans un thread daemon."""
        if self._thread and self._thread.is_alive():
            print("[WatcherService] (start) déjà démarré.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

        print("[WatcherService] ✅ Service de surveillance démarré.")

    def stop(self) -> None:
        """Arrête la boucle."""
        self._stop_event.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass

    def run_forever(self) -> None:
        print("[WatcherService] Boucle de surveillance active.")
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                self.poll_once()
            except Exception as e:
                print("[WatcherService] Erreur poll_once:", e)

            elapsed = time.time() - t0
            sleep_s = max(2.0, float(self.poll_interval_sec) - elapsed)
            # attend (interruptible)
            self._stop_event.wait(sleep_s)

    # ------------- core -------------

    def poll_once(self) -> None:
        sources = self._load_registry_sources()

        active_sources = [s for s in sources if s.active and s.url]
        if not active_sources:
            now = time.time()
            if now - self._last_no_sources_log > 30:  # log au max toutes les 30s
                print("[WatcherService] Aucune source active dans le registry.")
                self._last_no_sources_log = now
            return

        # On poll chaque source (RSS)
        for src in active_sources:
            if self._stop_event.is_set():
                return
            try:
                if (src.type or "").lower() == "rss":
                    self._poll_rss(src)
                else:
                    # tu peux rajouter d'autres types plus tard ("page", "pdf", etc.)
                    pass
            except Exception as e:
                print(f"[WatcherService] Source error ({src.url}):", e)

    def _poll_rss(self, src: WatchSource) -> None:
        headers = {"User-Agent": self.user_agent}
        r = requests.get(src.url, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()

        feed = feedparser.parse(r.content)
        entries = getattr(feed, "entries", []) or []

        # petite info utile en debug
        # print(f"[WatcherService] RSS {src.url} -> {len(entries)} items")

        for e in entries:
            if self._stop_event.is_set():
                return

            title = (getattr(e, "title", "") or "").strip()
            link = (getattr(e, "link", "") or "").strip()

            if not link:
                continue

            canon = canonicalize_url(link)
            site = get_site_from_url(canon) or get_site_from_url(link) or (src.label or "")

            published = ""
            # feedparser fournit souvent published / updated, on garde textuel (simple)
            published = (getattr(e, "published", "") or "").strip() or (getattr(e, "updated", "") or "").strip()

            # insert en base si nouveau
            is_new = self._db_insert_if_new(
                url=link,
                canonical_url=canon,
                title=title,
                site=site,
                published_at=published,
                source_url=src.url,
                entity_id=src.entity_id
            )

            if is_new:
                # callback UI (ton onglet News)
                self._emit_read(site=site, title=title, url=canon or link)

    def _emit_read(self, site: str, title: str, url: str) -> None:
        if callable(self.on_read):
            try:
                self.on_read(site, title, url)
            except Exception:
                # ne pas casser la boucle si l'UI a un souci
                pass

    # ------------- registry -------------

    def _load_registry_sources(self) -> List[WatchSource]:
        """
        Supporte un registry JSON souple :
        - top-level "sources": [...]
        - ou top-level "entities": [{..., "sources":[...]}]
        Chaque source peut être:
          {"url": "...", "type":"rss", "active": true, "label":"...", "entity_id":"..."}
        """
        path = self.registry_path

        if not os.path.exists(path):
            # pas d'erreur, juste aucune source
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("[WatcherService] Registry illisible:", e)
            return []

        out: List[WatchSource] = []

        # 1) sources globales
        if isinstance(data, dict) and isinstance(data.get("sources"), list):
            out.extend(self._parse_sources_list(data["sources"], default_entity_id=""))

        # 2) sources par entité
        if isinstance(data, dict) and isinstance(data.get("entities"), list):
            for ent in data["entities"]:
                if not isinstance(ent, dict):
                    continue
                ent_id = (ent.get("entity_id") or ent.get("id") or "").strip()
                sources = ent.get("sources")
                if isinstance(sources, list):
                    out.extend(self._parse_sources_list(sources, default_entity_id=ent_id))

        # si rien trouvé mais data est déjà une liste -> tolérance
        if not out and isinstance(data, list):
            out.extend(self._parse_sources_list(data, default_entity_id=""))

        # petite normalisation
        cleaned = []
        for s in out:
            if not s.url:
                continue
            if not s.type:
                s.type = "rss"
            cleaned.append(s)
        return cleaned

    def _parse_sources_list(self, sources: List[Any], default_entity_id: str = "") -> List[WatchSource]:
        out: List[WatchSource] = []
        for s in sources:
            if isinstance(s, str):
                out.append(WatchSource(url=s, type="rss", active=True, entity_id=default_entity_id))
                continue
            if not isinstance(s, dict):
                continue

            url = (s.get("url") or s.get("link") or "").strip()
            if not url:
                continue

            stype = (s.get("type") or "rss").strip().lower()
            active = bool(s.get("active", True))
            label = (s.get("label") or s.get("name") or "").strip()
            entity_id = (s.get("entity_id") or default_entity_id or "").strip()

            poll = 0
            try:
                poll = int(s.get("poll_interval_sec") or 0)
            except Exception:
                poll = 0

            out.append(WatchSource(
                url=url,
                type=stype,
                active=active,
                label=label,
                entity_id=entity_id,
                poll_interval_sec=poll,
            ))
        return out

    # ------------- sqlite -------------

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS news_reads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetched_at TEXT NOT NULL,
                    source_url TEXT,
                    entity_id TEXT,
                    site TEXT,
                    title TEXT,
                    url TEXT NOT NULL,
                    canonical_url TEXT,
                    published_at TEXT
                )
            """)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_news_reads_url ON news_reads(url)")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_news_reads_canon ON news_reads(canonical_url)")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_news_reads_fetched_at ON news_reads(fetched_at)")
            con.commit()
        finally:
            con.close()

    def _db_insert_if_new(
        self,
        url: str,
        canonical_url: str,
        title: str,
        site: str,
        published_at: str,
        source_url: str,
        entity_id: str
    ) -> bool:
        """
        Retourne True si c'était nouveau (donc inséré), False si déjà vu.
        """
        fetched_at = time.strftime("%Y-%m-%d %H:%M:%S")
        url = (url or "").strip()
        canonical_url = (canonical_url or "").strip()

        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()

            # stratégie : on tente d’insérer, si contrainte unique -> déjà vu
            try:
                cur.execute("""
                    INSERT INTO news_reads (fetched_at, source_url, entity_id, site, title, url, canonical_url, published_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (fetched_at, source_url, entity_id, site, title, url, canonical_url, published_at))
                con.commit()
                return True
            except sqlite3.IntegrityError:
                return False
        finally:
            con.close()
