# micheline/intel/event_cards.py
# Bloc 3 — Normalisation en "Event Cards" (JSON) + stockage SQLite séparé
# Objectif: transformer raw_events -> objets actionnables (type, entités, claims, scores)

from __future__ import annotations

import os
import re
import json
import uuid
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import config


EVENT_CARDS_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "db",
    "event_cards.sqlite",
)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "[]"


def _strip_html(text: str) -> str:
    text = text or ""
    # retire tags HTML basique
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------
# DB: Event Cards
# --------------------------

class EventCardsDB:
    """
    DB séparée pour éviter les verrous avec chat/mémoire.
    1 card par raw_event (unique sur raw_event_hash).
    """

    def __init__(self, db_path: str = EVENT_CARDS_DB_PATH):
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
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_cards (
                    card_id TEXT PRIMARY KEY,

                    raw_event_id TEXT,
                    raw_event_hash TEXT NOT NULL UNIQUE,

                    first_seen_at TEXT NOT NULL,
                    source_type TEXT,

                    urls TEXT,
                    entities TEXT,

                    event_type TEXT,
                    claims TEXT,

                    novelty_score REAL,
                    severity_score REAL,
                    evidence_score REAL,

                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_cards_first_seen ON event_cards(first_seen_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_cards_type ON event_cards(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_cards_evidence ON event_cards(evidence_score DESC)")
            conn.commit()

    def insert_if_new(self, card: Dict[str, Any]) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO event_cards (
                        card_id,
                        raw_event_id, raw_event_hash,
                        first_seen_at, source_type,
                        urls, entities,
                        event_type, claims,
                        novelty_score, severity_score, evidence_score,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.get("card_id"),
                        card.get("raw_event_id"),
                        card.get("raw_event_hash"),
                        card.get("first_seen_at"),
                        card.get("source_type"),
                        card.get("urls"),
                        card.get("entities"),
                        card.get("event_type"),
                        card.get("claims"),
                        float(card.get("novelty_score", 0.0)),
                        float(card.get("severity_score", 0.0)),
                        float(card.get("evidence_score", 0.0)),
                        card.get("created_at"),
                    ),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"[EventCardsDB] Erreur insertion: {e}")
            return False

    def purge_older_than_days(self, days: int = 7) -> int:
        try:
            days = int(days)
            if days <= 0:
                return 0
        except Exception:
            days = 7

        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM event_cards WHERE first_seen_at < datetime('now', ?)",
                    (f"-{days} days",),
                )
                deleted = cur.rowcount if cur.rowcount is not None else 0
                conn.commit()
                if deleted:
                    print(f"[EventCardsDB] Purge: {deleted} card(s) supprimée(s) (> {days} jours)")
                return int(deleted)
        except Exception as e:
            print(f"[EventCardsDB] Purge erreur: {e}")
            return 0


# --------------------------
# Normalizer (Bloc 3)
# --------------------------

class EventCardNormalizer:
    """
    Normalise un raw_event -> Event Card
    - event_type: heuristique keywords (simple mais efficace)
    - entities: entity_id/name + détections complémentaires
    - claims: "qui dit quoi" (speaker=site/entity + extrait)
    - scores: novelty/severity/evidence (heuristiques)
    """

    # Heuristique de types
    EVENT_TYPE_RULES = [
        ("central_bank_signal", [
            "ecb", "bce", "fomc", "federal reserve", "fed",
            "interest rate", "rate hike", "rate cut", "policy rate",
            "hausse de taux", "baisse de taux", "taux directeur", "banque centrale",
            "monetary policy", "inflation target"
        ]),
        ("sanctions", [
            "sanction", "embargo", "export ban", "export controls", "blacklist",
            "asset freeze", "designated", "ofac", "swift ban"
        ]),
        ("military_escalation", [
            "attack", "strike", "airstrike", "missile", "drone", "bomb", "shelling",
            "invasion", "troops", "escalation", "ceasefire", "mobilization",
            "attaque", "frappe", "missile", "drone", "bombard", "invasion", "troupes"
        ]),
        ("shipping_accident", [
            "ship", "boat", "vessel", "tanker", "cargo", "capsize", "sank", "maritime",
            "navire", "bateau", "pétrolier", "cargo", "naufrage", "échoué"
        ]),
        ("commodity_supply", [
            "opec", "opec+", "oil output", "production cut", "barrel", "brent", "wti",
            "pétrole", "baril", "production", "quota"
        ]),
        ("macro_data", [
            "cpi", "inflation", "gdp", "unemployment", "pmi", "jobs report", "nfp",
            "indice des prix", "croissance", "chômage", "pib", "pmi"
        ]),
        ("market_move", [
            "stocks", "shares", "bond yields", "treasury", "sell-off", "rally",
            "bourse", "actions", "obligations", "rendements", "chute", "hausse"
        ]),
        ("odd_news", [
            "kangaroo", "zoo", "escaped", "escape", "animal",
            "kangourou", "zoo", "évadé", "s'évade", "animal"
        ]),
    ]

    # Gravité par type (0-1)
    SEVERITY_BY_TYPE = {
        "military_escalation": 0.90,
        "sanctions": 0.75,
        "central_bank_signal": 0.70,
        "commodity_supply": 0.65,
        "macro_data": 0.55,
        "market_move": 0.45,
        "shipping_accident": 0.35,
        "odd_news": 0.10,
        "unknown": 0.25,
    }

    # Qualité “source_type” => evidence_score de base
    EVIDENCE_BY_SOURCE_TYPE = {
        "official_doc": 0.85,
        "rss": 0.65,
        "website": 0.55,
        "social": 0.40,
    }

    # Domaines “souvent fiables” (tu peux étendre)
    HIGH_TRUST_DOMAINS = {
        "www.ecb.europa.eu",
        "ecb.europa.eu",
        "www.federalreserve.gov",
        "federalreserve.gov",
        "www.imf.org",
        "imf.org",
        "www.bis.org",
        "bis.org",
        "www.worldbank.org",
        "worldbank.org",
        "www.opec.org",
        "opec.org",
    }

    def normalize(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        title = (raw_event.get("title") or "").strip()
        content = (raw_event.get("content") or "").strip()
        content_txt = _strip_html(content)

        url = (raw_event.get("url") or raw_event.get("source_url") or "").strip()
        source_type = (raw_event.get("source_type") or "").strip().lower()

        # metadata JSON (optionnel)
        meta = {}
        try:
            meta = json.loads(raw_event.get("metadata") or "{}")
        except Exception:
            meta = {}

        entity_id = (raw_event.get("entity_id") or "").strip()
        entity_name = (meta.get("entity_name") or "").strip()

        text = (title + "\n" + content_txt).lower()

        event_type = self._classify_event_type(text=text, entity_id=entity_id, entity_name=entity_name, url=url)
        entities = self._extract_entities(text=text, entity_id=entity_id, entity_name=entity_name)
        claims = self._make_claims(site=_domain(url), entity_name=entity_name, title=title, content_txt=content_txt, url=url)

        novelty = 1.0  # à ce stade, 1 raw_event = 1 nouveau signal (le clustering viendra bloc 4)
        severity = float(self.SEVERITY_BY_TYPE.get(event_type, 0.25))
        evidence = self._compute_evidence_score(
            source_type=source_type,
            url=url,
            trust_score=raw_event.get("trust_score", None),  # souvent absent dans raw_event
        )

        card = {
            "card_id": str(uuid.uuid4()),
            "raw_event_id": raw_event.get("event_id"),
            "raw_event_hash": raw_event.get("content_hash"),

            "first_seen_at": raw_event.get("fetched_at") or _now_str(),
            "source_type": source_type,

            "urls": _json([u for u in [url] if u]),
            "entities": _json(entities),

            "event_type": event_type,
            "claims": _json(claims),

            "novelty_score": novelty,
            "severity_score": severity,
            "evidence_score": evidence,

            "created_at": _now_str(),
        }
        return card

    def _classify_event_type(self, text: str, entity_id: str, entity_name: str, url: str) -> str:
        # 1) keywords rules
        for etype, keywords in self.EVENT_TYPE_RULES:
            for k in keywords:
                if k in text:
                    return etype

        # 2) fallback par entity_name
        en = (entity_name or "").lower()
        if en in ("european central bank", "ecb", "banque centrale européenne"):
            return "central_bank_signal"
        if "opec" in en:
            return "commodity_supply"

        # 3) fallback par domaine
        dom = _domain(url)
        if dom.endswith("ecb.europa.eu") or dom.endswith("federalreserve.gov"):
            return "central_bank_signal"

        return "unknown"

    def _extract_entities(self, text: str, entity_id: str, entity_name: str) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        if entity_id or entity_name:
            entities.append({
                "entity_id": entity_id or None,
                "name": entity_name or None,
                "role": "primary",
            })

        # détections simples (tu peux enrichir)
        keyword_entities = [
            ("Iran", ["iran", "iranian", "téhéran", "tehran"]),
            ("Israel", ["israel", "israeli", "tel aviv", "gaza"]),
            ("Russia", ["russia", "russian", "moscow", "ukraine", "ukrainian", "kyiv", "kiev"]),
            ("China", ["china", "chinese", "beijing", "taiwan", "taipei"]),
            ("United States", ["united states", "u.s.", "usa", "washington"]),
            ("Oil", ["oil", "brent", "wti", "barrel", "pétrole", "baril"]),
            ("Rates", ["interest rate", "policy rate", "taux", "rate hike", "rate cut", "hausse de taux", "baisse de taux"]),
        ]

        for name, kws in keyword_entities:
            for k in kws:
                if k in text:
                    entities.append({"entity_id": None, "name": name, "role": "detected"})
                    break

        # dédup par name
        seen = set()
        uniq = []
        for e in entities:
            key = (e.get("entity_id") or "") + "|" + (e.get("name") or "")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        return uniq

    def _make_claims(self, site: str, entity_name: str, title: str, content_txt: str, url: str) -> List[Dict[str, Any]]:
        speaker = entity_name or site or "source"
        excerpt = (content_txt or "")
        excerpt = excerpt[:350].strip()
        if not excerpt:
            excerpt = (title or "")[:350].strip()

        claim_text = title.strip()
        if excerpt and excerpt.lower() not in claim_text.lower():
            claim_text = (claim_text + " — " + excerpt).strip()

        return [{
            "speaker": speaker,
            "text": claim_text[:500],
            "url": url,
        }]

    def _compute_evidence_score(self, source_type: str, url: str, trust_score: Any = None) -> float:
        base = float(self.EVIDENCE_BY_SOURCE_TYPE.get(source_type or "rss", 0.55))
        dom = _domain(url)

        if dom in self.HIGH_TRUST_DOMAINS:
            base = max(base, 0.80)

        # si trust_score est fourni (rare dans raw_event), on l'injecte
        try:
            if trust_score is not None:
                ts = float(trust_score)
                base = max(base, min(1.0, ts))
        except Exception:
            pass

        # clamp
        return max(0.0, min(1.0, base))