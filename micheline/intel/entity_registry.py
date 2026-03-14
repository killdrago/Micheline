# micheline/intel/entity_registry.py
# Registry centralisée des entités surveillées (personnes/institutions/pays) + sources
# - Auto-discovery via LLM (avec validation par règles)
# - Scoring d'importance basé sur récurrence multi-sources

import os
import json
import sqlite3
from typing import List, Dict, Optional, Set
from datetime import datetime
from pathlib import Path

# Chemin de la base SQLite du registry (séparée de l'historique chat)
REGISTRY_DB_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "intel", "db", "entity_registry.sqlite"
)

class EntityRegistry:
    """
    Gère la base de connaissances structurée des entités surveillées.
    Permet l'auto-discovery contrôlée (proposition LLM → validation règles).
    """
    
    def __init__(self, db_path: str = REGISTRY_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row  # Résultats accessibles par nom de colonne
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
            
            # Table principale des entités
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                aliases TEXT,  -- JSON array: ["Donald Trump", "Trump", "@realDonaldTrump"]
                entity_type TEXT NOT NULL,  -- person/institution/country/asset
                importance_score REAL DEFAULT 0.5,  -- 0.0-1.0
                topics TEXT,  -- JSON array: ["geo", "oil", "rates", "fx"]
                created_at TEXT NOT NULL,
                last_updated_at TEXT NOT NULL,
                notes TEXT
            )
            """)
            
            # Table des sources associées à chaque entité
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                source_type TEXT NOT NULL,  -- rss/website/social/official_doc
                url TEXT NOT NULL,
                trust_score REAL DEFAULT 0.5,  -- 0.0-1.0 (backlinks, concordance)
                first_seen_at TEXT NOT NULL,
                last_verified_at TEXT,
                is_active INTEGER DEFAULT 1,  -- 0/1
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            )
            """)
            
            # Index pour recherches rapides
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type 
            ON entities(entity_type)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_importance 
            ON entities(importance_score DESC)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sources_entity 
            ON entity_sources(entity_id)
            """)
            
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sources_active 
            ON entity_sources(is_active)
            """)
            
            conn.commit()
    
    # ========== CRUD Entités ==========
    
    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        importance_score: float = 0.5,
        notes: str = ""
    ) -> bool:
        """
        Ajoute une nouvelle entité au registry.
        Retourne True si succès, False si l'entité existe déjà.
        """
        try:
            with self._get_conn() as conn:
                now = datetime.now().isoformat()
                
                conn.execute("""
                INSERT INTO entities (
                    entity_id, name, aliases, entity_type, 
                    importance_score, topics, created_at, last_updated_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity_id,
                    name,
                    json.dumps(aliases or []),
                    entity_type,
                    max(0.0, min(1.0, importance_score)),
                    json.dumps(topics or []),
                    now,
                    now,
                    notes
                ))
                conn.commit()
                print(f"[Registry] Entité ajoutée: {name} ({entity_id})")
                return True
        except sqlite3.IntegrityError:
            print(f"[Registry] Entité {entity_id} existe déjà.")
            return False
        except Exception as e:
            print(f"[Registry] Erreur ajout entité: {e}")
            return False
    
    def update_entity_importance(self, entity_id: str, new_score: float) -> bool:
        """Met à jour le score d'importance (basé sur récurrence multi-sources)."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                UPDATE entities 
                SET importance_score = ?, last_updated_at = ?
                WHERE entity_id = ?
                """, (
                    max(0.0, min(1.0, new_score)),
                    datetime.now().isoformat(),
                    entity_id
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"[Registry] Erreur mise à jour importance: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Récupère les infos complètes d'une entité."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                SELECT * FROM entities WHERE entity_id = ?
                """, (entity_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    "entity_id": row["entity_id"],
                    "name": row["name"],
                    "aliases": json.loads(row["aliases"] or "[]"),
                    "entity_type": row["entity_type"],
                    "importance_score": row["importance_score"],
                    "topics": json.loads(row["topics"] or "[]"),
                    "created_at": row["created_at"],
                    "last_updated_at": row["last_updated_at"],
                    "notes": row["notes"]
                }
        except Exception as e:
            print(f"[Registry] Erreur récupération entité: {e}")
            return None
    
    def list_entities(
        self,
        entity_type: Optional[str] = None,
        min_importance: float = 0.0,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        """Liste les entités avec filtres optionnels."""
        try:
            with self._get_conn() as conn:
                query = "SELECT * FROM entities WHERE importance_score >= ?"
                params = [min_importance]
                
                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type)
                
                query += " ORDER BY importance_score DESC"
                
                cursor = conn.execute(query, params)
                entities = []
                
                for row in cursor.fetchall():
                    entity = {
                        "entity_id": row["entity_id"],
                        "name": row["name"],
                        "aliases": json.loads(row["aliases"] or "[]"),
                        "entity_type": row["entity_type"],
                        "importance_score": row["importance_score"],
                        "topics": json.loads(row["topics"] or "[]"),
                        "created_at": row["created_at"],
                        "last_updated_at": row["last_updated_at"],
                        "notes": row["notes"]
                    }
                    
                    # Filtre par topics si spécifié
                    if topics:
                        entity_topics = set(entity["topics"])
                        if not entity_topics.intersection(topics):
                            continue
                    
                    entities.append(entity)
                
                return entities
        except Exception as e:
            print(f"[Registry] Erreur listing entités: {e}")
            return []
    
    # ========== CRUD Sources ==========
    
    def add_source(
        self,
        entity_id: str,
        source_type: str,
        url: str,
        trust_score: float = 0.5
    ) -> bool:
        """Associe une nouvelle source à une entité."""
        try:
            with self._get_conn() as conn:
                # Vérifie que l'entité existe
                cursor = conn.execute(
                    "SELECT 1 FROM entities WHERE entity_id = ?",
                    (entity_id,)
                )
                if not cursor.fetchone():
                    print(f"[Registry] Entité {entity_id} introuvable.")
                    return False
                
                now = datetime.now().isoformat()
                
                conn.execute("""
                INSERT INTO entity_sources (
                    entity_id, source_type, url, trust_score, 
                    first_seen_at, last_verified_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (
                    entity_id,
                    source_type,
                    url,
                    max(0.0, min(1.0, trust_score)),
                    now,
                    now
                ))
                conn.commit()
                print(f"[Registry] Source ajoutée: {url} → {entity_id}")
                return True
        except Exception as e:
            print(f"[Registry] Erreur ajout source: {e}")
            return False
    
    def update_source_trust(self, source_id: int, new_trust: float) -> bool:
        """Met à jour le score de confiance d'une source (backlinks, concordance)."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                UPDATE entity_sources 
                SET trust_score = ?, last_verified_at = ?
                WHERE source_id = ?
                """, (
                    max(0.0, min(1.0, new_trust)),
                    datetime.now().isoformat(),
                    source_id
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"[Registry] Erreur mise à jour trust: {e}")
            return False
    
    def get_entity_sources(
        self,
        entity_id: str,
        active_only: bool = True
    ) -> List[Dict]:
        """Récupère toutes les sources associées à une entité."""
        try:
            with self._get_conn() as conn:
                query = "SELECT * FROM entity_sources WHERE entity_id = ?"
                params = [entity_id]
                
                if active_only:
                    query += " AND is_active = 1"
                
                query += " ORDER BY trust_score DESC"
                
                cursor = conn.execute(query, params)
                sources = []
                
                for row in cursor.fetchall():
                    sources.append({
                        "source_id": row["source_id"],
                        "entity_id": row["entity_id"],
                        "source_type": row["source_type"],
                        "url": row["url"],
                        "trust_score": row["trust_score"],
                        "first_seen_at": row["first_seen_at"],
                        "last_verified_at": row["last_verified_at"],
                        "is_active": bool(row["is_active"])
                    })
                
                return sources
        except Exception as e:
            print(f"[Registry] Erreur récupération sources: {e}")
            return []
    
    def list_all_active_sources(self) -> List[Dict]:
        """Liste toutes les sources actives (pour le watcher)."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                SELECT s.*, e.name as entity_name, e.importance_score
                FROM entity_sources s
                JOIN entities e ON s.entity_id = e.entity_id
                WHERE s.is_active = 1
                ORDER BY e.importance_score DESC, s.trust_score DESC
                """)
                
                sources = []
                for row in cursor.fetchall():
                    sources.append({
                        "source_id": row["source_id"],
                        "entity_id": row["entity_id"],
                        "entity_name": row["entity_name"],
                        "entity_importance": row["importance_score"],
                        "source_type": row["source_type"],
                        "url": row["url"],
                        "trust_score": row["trust_score"],
                        "last_verified_at": row["last_verified_at"]
                    })
                
                return sources
        except Exception as e:
            print(f"[Registry] Erreur listing sources actives: {e}")
            return []
    
    # ========== Auto-Discovery (propositions LLM → validation) ==========
    
    def propose_entity(
        self,
        name: str,
        entity_type: str,
        proposed_sources: List[str],
        topics: Optional[List[str]] = None,
        llm_reasoning: str = ""
    ) -> Dict:
        """
        Enregistre une proposition d'entité (issue du LLM).
        La promotion en entité active nécessite validation par règles.
        
        Retourne:
        {
            "status": "pending_validation" | "auto_approved" | "rejected",
            "entity_id": str (si approuvé),
            "reason": str
        }
        """
        # TODO: Implémenter la logique de validation (règles de recoupement)
        # Pour l'instant, on log juste la proposition
        
        print(f"[Registry] Proposition LLM: {name} ({entity_type})")
        print(f"  Sources proposées: {proposed_sources}")
        print(f"  Reasoning: {llm_reasoning}")
        
        # Règles de base pour auto-approbation (à affiner) :
        # 1. Au moins 2 sources distinctes
        # 2. Sources contiennent des domaines reconnus (ex: .gov, médias majeurs)
        # 3. Pas déjà dans le registry
        
        entity_id_candidate = name.lower().replace(" ", "_")
        
        if self.get_entity(entity_id_candidate):
            return {
                "status": "rejected",
                "entity_id": None,
                "reason": "Entité déjà existante"
            }
        
        if len(proposed_sources) < 2:
            return {
                "status": "pending_validation",
                "entity_id": None,
                "reason": "Insuffisant: nécessite au moins 2 sources distinctes"
            }
        
        # Auto-approval basique (à renforcer avec backlink checker)
        trusted_domains = {".gov", ".mil", "reuters.com", "bloomberg.com", "ft.com"}
        has_trusted = any(
            any(domain in url.lower() for domain in trusted_domains)
            for url in proposed_sources
        )
        
        if has_trusted and len(proposed_sources) >= 2:
            # Auto-approve
            success = self.add_entity(
                entity_id=entity_id_candidate,
                name=name,
                entity_type=entity_type,
                topics=topics or [],
                importance_score=0.6,  # Score initial modéré
                notes=f"Auto-discovered via LLM. Reasoning: {llm_reasoning}"
            )
            
            if success:
                # Ajoute les sources
                for url in proposed_sources:
                    self.add_source(
                        entity_id=entity_id_candidate,
                        source_type="website",  # À affiner selon l'URL
                        url=url,
                        trust_score=0.7 if any(d in url.lower() for d in trusted_domains) else 0.5
                    )
                
                return {
                    "status": "auto_approved",
                    "entity_id": entity_id_candidate,
                    "reason": "Sources fiables + recoupement suffisant"
                }
        
        return {
            "status": "pending_validation",
            "entity_id": None,
            "reason": "Nécessite validation manuelle (sources non suffisamment fiables)"
        }
    
    # ========== Helpers de recherche ==========
    
    def find_entities_by_alias(self, alias_query: str) -> List[Dict]:
        """Recherche des entités par nom ou alias (fuzzy)."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                SELECT * FROM entities 
                WHERE name LIKE ? 
                   OR aliases LIKE ?
                ORDER BY importance_score DESC
                """, (f"%{alias_query}%", f"%{alias_query}%"))
                
                entities = []
                for row in cursor.fetchall():
                    entities.append({
                        "entity_id": row["entity_id"],
                        "name": row["name"],
                        "aliases": json.loads(row["aliases"] or "[]"),
                        "entity_type": row["entity_type"],
                        "importance_score": row["importance_score"],
                        "topics": json.loads(row["topics"] or "[]")
                    })
                
                return entities
        except Exception as e:
            print(f"[Registry] Erreur recherche alias: {e}")
            return []
    
    def get_critical_entities(self, threshold: float = 0.7) -> List[Dict]:
        """Retourne les entités considérées comme critiques (importance >= threshold)."""
        return self.list_entities(min_importance=threshold)


# ========== Fonction d'initialisation (seed data) ==========

def seed_default_entities():
    """
    Peuple le registry avec des entités critiques par défaut.
    À appeler une seule fois au setup ou manuellement.
    """
    registry = EntityRegistry()
    
    # Exemples d'entités critiques (à adapter selon tes besoins)
    default_entities = [
        {
            "entity_id": "donald_trump",
            "name": "Donald Trump",
            "entity_type": "person",
            "aliases": ["Trump", "Donald J. Trump", "@realDonaldTrump"],
            "topics": ["geo", "us_politics", "tariffs", "fx"],
            "importance_score": 0.9,
            "sources": [
                ("social", "https://truthsocial.com/@realDonaldTrump", 0.9),
                # RSS public de suivi (exemple)
                ("rss", "https://rsshub.app/truthsocial/realDonaldTrump", 0.8),
            ],
            "notes": "45e et 47e président des États-Unis. Impact majeur sur marchés via annonces."
        },
        {
            "entity_id": "federal_reserve",
            "name": "Federal Reserve",
            "entity_type": "institution",
            "aliases": ["Fed", "US Federal Reserve", "FOMC"],
            "topics": ["rates", "monetary_policy", "fx", "bonds"],
            "importance_score": 1.0,
            "sources": [
                ("official_doc", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", 1.0),
                ("rss", "https://www.federalreserve.gov/feeds/press_all.xml", 0.95),
            ],
            "notes": "Banque centrale US. Décisions de taux = impact critique sur USD et actifs globaux."
        },
        {
            "entity_id": "iran",
            "name": "Iran",
            "entity_type": "country",
            "aliases": ["Islamic Republic of Iran", "Tehran"],
            "topics": ["geo", "oil", "sanctions", "middle_east"],
            "importance_score": 0.85,
            "sources": [
                ("website", "https://en.irna.ir/", 0.7),  # Agence de presse officielle
                ("rss", "https://www.aljazeera.com/xml/rss/all.xml", 0.75),  # Couverture régionale
            ],
            "notes": "Géopolitique ME. Tensions → pétrole, sanctions → USD/IRR."
        },
        {
            "entity_id": "opec",
            "name": "OPEC",
            "entity_type": "institution",
            "aliases": ["Organization of Petroleum Exporting Countries"],
            "topics": ["oil", "energy", "commodities"],
            "importance_score": 0.9,
            "sources": [
                ("official_doc", "https://www.opec.org/opec_web/en/press_room/", 0.95),
            ],
            "notes": "Décisions de production → pétrole WTI/Brent."
        },
        {
            "entity_id": "ecb",
            "name": "European Central Bank",
            "entity_type": "institution",
            "aliases": ["ECB", "BCE"],
            "topics": ["rates", "monetary_policy", "fx", "eur"],
            "importance_score": 0.95,
            "sources": [
                ("official_doc", "https://www.ecb.europa.eu/press/calendars/html/index.en.html", 1.0),
                ("rss", "https://www.ecb.europa.eu/rss/press.html", 0.95),
            ],
            "notes": "Banque centrale européenne. Impact EUR."
        },
        {
            "entity_id": "french_markets",
            "name": "Marchés Français",
            "entity_type": "asset",
            "aliases": ["CAC40", "Bourse Paris"],
            "topics": ["fx", "stocks", "european_markets"],
            "importance_score": 0.75,
            "sources": [
                ("rss", "https://www.boursorama.com/bourse/actualites/rss/", 0.75),
                ("rss", "https://fr.investing.com/rss/news.rss", 0.70),
                ("website", "https://www.boursier.com/actualites", 0.70),
            ],
            "notes": "Actualités marchés financiers français"
        },
        {
            "entity_id": "banque_de_france",
            "name": "Banque de France",
            "entity_type": "institution",
            "aliases": ["BdF", "Banque France"],
            "topics": ["rates", "monetary_policy", "eur", "french_economy"],
            "importance_score": 0.80,
            "sources": [
                ("website", "https://www.banque-france.fr/communiques-de-presse", 0.90),
                ("official_doc", "https://www.banque-france.fr/statistiques", 0.85),
            ],
            "notes": "Banque centrale française (taux, politique monétaire)"
        },
    ]
    
    for entity_data in default_entities:
        # Ajoute l'entité
        success = registry.add_entity(
            entity_id=entity_data["entity_id"],
            name=entity_data["name"],
            entity_type=entity_data["entity_type"],
            aliases=entity_data.get("aliases", []),
            topics=entity_data.get("topics", []),
            importance_score=entity_data.get("importance_score", 0.5),
            notes=entity_data.get("notes", "")
        )
        
        if success:
            # Ajoute les sources
            for source_type, url, trust in entity_data.get("sources", []):
                registry.add_source(
                    entity_id=entity_data["entity_id"],
                    source_type=source_type,
                    url=url,
                    trust_score=trust
                )
    
    print("[Registry] Seed data : entités par défaut ajoutées.")

    def seed_news_portfolio_sources():
        """
        Ajoute (si manquantes) des sources RSS "grand éventail" dans le registry.
        Idempotent: ne crée pas de doublons.
        """
        registry = EntityRegistry()

        entity_id = "news_portfolio"
        # Crée l'entité si elle n'existe pas (si elle existe, add_entity retourne False, ce n'est pas grave)
        registry.add_entity(
            entity_id=entity_id,
            name="News Portfolio (RSS Radar)",
            entity_type="institution",
            aliases=["news_portfolio", "rss_radar"],
            topics=["news", "world", "fr", "markets", "rates", "oil"],
            importance_score=0.85,
            notes="Portfolio de sources RSS (officielles + Google News RSS par site) pour couvrir large."
        )

        # URLs déjà présentes (pour éviter les doublons)
        existing = set()
        try:
            for s in registry.get_entity_sources(entity_id, active_only=False):
                existing.add((s.get("source_type", ""), s.get("url", "")))
        except Exception:
            pass

        def gnews_site(site: str, hl="fr", gl="FR", ceid="FR:fr") -> str:
            # Google News RSS "site:" (très pratique quand le site n'a pas de RSS public fiable)
            # Note: ":" encodé en %3A
            return f"https://news.google.com/rss/search?q=site%3A{site}&hl={hl}&gl={gl}&ceid={ceid}"

        sources = [
            # ---------- RSS officiels (fiables) ----------
            ("rss", "https://www.lemonde.fr/rss/en_continu.xml", 0.80),
            ("rss", "https://www.lemonde.fr/economie/rss_full.xml", 0.75),
            ("rss", "https://www.lemonde.fr/international/rss_full.xml", 0.75),

            ("rss", "http://feeds.bbci.co.uk/news/world/rss.xml", 0.75),
            ("rss", "https://www.aljazeera.com/xml/rss/all.xml", 0.70),

            # France Diplomatie (RSS officiel)
            ("rss", "http://www.diplomatie.gouv.fr/spip.php?page=backend-fd", 0.80),

            # Banques centrales / institutions (RSS officiels)
            ("rss", "https://www.ecb.europa.eu/rss/press.html", 0.90),
            ("rss", "https://www.ecb.europa.eu/rss/blog.html", 0.80),
            ("rss", "https://www.bis.org/doclist/all_pressrels.rss", 0.85),
            ("rss", "https://www.bankofengland.co.uk/rss/news", 0.80),

            # RBA (URLs RSS révélées par leur page RSS)
            ("rss", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml", 0.75),

            # ---------- Sites “difficiles” => Google News RSS par site ----------
            ("rss", gnews_site("boursorama.com"), 0.60),
            ("rss", gnews_site("boursier.com"), 0.60),
            ("rss", gnews_site("investing.com"), 0.60),
            ("rss", gnews_site("lefigaro.fr"), 0.55),
            ("rss", gnews_site("lesechos.fr"), 0.60),

            # Fed: chez toi ça se fait bloquer par robots sur certains chemins => Google News RSS par site
            ("rss", gnews_site("federalreserve.gov", hl="en", gl="US", ceid="US:en"), 0.70),
        ]

        added = 0
        for source_type, url, trust in sources:
            key = (source_type, url)
            if key in existing:
                continue
            ok = registry.add_source(entity_id=entity_id, source_type=source_type, url=url, trust_score=trust)
            if ok:
                existing.add(key)
                added += 1

        print(f"[Registry] seed_news_portfolio_sources: +{added} source(s) ajoutée(s).")

# ========== Point d'entrée CLI (optionnel) ==========

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--seed":
        print("Initialisation du registry avec les entités par défaut...")
        seed_default_entities()
        print("Terminé.")
    else:
        # Exemple d'utilisation
        registry = EntityRegistry()
        
        print("\n=== Entités critiques (importance >= 0.7) ===")
        for entity in registry.get_critical_entities():
            print(f"  • {entity['name']} ({entity['entity_type']}) — score: {entity['importance_score']}")
        
        print("\n=== Sources actives à surveiller ===")
        for source in registry.list_all_active_sources()[:10]:  # Top 10
            print(f"  • {source['entity_name']}: {source['url']}")
