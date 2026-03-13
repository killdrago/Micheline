# micheline/rag/vector_store.py
# Gère la base de connaissances vectorielle avec FAISS et sentence-transformers.

import os
from pathlib import Path
from typing import List, Dict
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from micheline.rag.document_loader import Document
import config  # utilisation des réglages centralisés

# Chemins par défaut (via config)
EMBEDDING_MODEL_DIR = Path(config.RAG_EMBEDDING_MODEL_DIR)
FAISS_INDEX_PATH = Path(config.RAG_FAISS_INDEX_PATH)

class KnowledgeBase:
    def __init__(self, index_path: str = str(FAISS_INDEX_PATH), embedding_model_path: str = str(EMBEDDING_MODEL_DIR)):
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix('.meta.json')
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("[KnowledgeBase] Initialisation...")
        self.embedding_model = self._load_embedding_model(embedding_model_path)
        self.index = self._load_faiss_index()
        self.metadata_store = self._load_metadata()

    def _load_embedding_model(self, model_path: str):
        """Charge le modèle d'embeddings local."""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Le dossier du modèle d'embeddings est introuvable: {model_path}")
            print(f"Chargement du modèle d'embeddings depuis: {model_path}")
            return SentenceTransformer(model_path)
        except Exception as e:
            print(f"[FATAL] Impossible de charger le modèle d'embeddings. Le RAG ne fonctionnera pas. Erreur: {e}")
            raise

    def _load_faiss_index(self):
        """Charge l'index FAISS depuis le disque ou en crée un nouveau."""
        if self.index_path.exists():
            print(f"Chargement de l'index FAISS existant: {self.index_path}")
            return faiss.read_index(str(self.index_path))
        else:
            print("Aucun index FAISS trouvé. Création d'un nouvel index.")
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            return faiss.IndexFlatL2(dimension)

    def _load_metadata(self):
        """Charge les métadonnées associées à l'index."""
        import json
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save(self):
        """Sauvegarde l'index FAISS et des métadonnées sur le disque."""
        import json
        print("Sauvegarde de l'index FAISS et des métadonnées...")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2)
        print("Sauvegarde terminée.")
        
    def _compact_if_needed(self):
        """Si l'index dépasse un seuil, le réduit à une taille cible en gardant les chunks les plus récents."""
        if not config.RAG_COMPACT_ENABLED:
            return

        trigger_size = int(config.RAG_COMPACT_TRIGGER_CHUNKS)
        target_size = int(config.RAG_COMPACT_TARGET_CHUNKS)
        current_size = self.index.ntotal

        if current_size <= trigger_size:
            return

        print(f"[RAG Compaction] Seuil dépassé ({current_size}/{trigger_size}). Compactage vers ~{target_size} chunks...")
        
        # Stratégie : garder les N derniers chunks ajoutés (les plus récents)
        num_to_remove = current_size - target_size
        if num_to_remove <= 0:
            return

        # Supprimer les plus anciens (indices de 0 à num_to_remove - 1)
        ids_to_remove = np.arange(num_to_remove)
        self.index.remove_ids(ids_to_remove)
        
        # Mettre à jour les métadonnées en supprimant les éléments correspondants au début
        self.metadata_store = self.metadata_store[num_to_remove:]
        
        print(f"[RAG Compaction] Compactage terminé. Nouvelle taille de l'index: {self.index.ntotal}")
        # La sauvegarde se fera à la fin de add_documents

    def add_documents(self, docs: List[Document]):
        """Encode et ajoute une liste de documents (chunks) à l'index."""
        if not docs:
            return
            
        print(f"Encodage de {len(docs)} chunks pour l'indexation...")
        contents = [doc.page_content for doc in docs]
        
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype='float32')

        # Ajoute les vecteurs à l'index FAISS et stocke les métadonnées
        self.index.add(embeddings)
        for doc in docs:
            self.metadata_store.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        print(f"Indexation terminée. Total de chunks dans la base: {self.index.ntotal}")

        # --- APPEL DU COMPACTAGE ---
        self._compact_if_needed()
        
        self._save()

    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        Recherche les k chunks les plus pertinents pour une requête donnée.
        Retourne une liste de dictionnaires contenant le contenu et les métadonnées.
        """
        if self.index.ntotal == 0:
            return []

        top_k = int(k or config.RAG_TOP_K)
        print(f"Recherche dans la base de connaissances pour la requête: '{query[:50]}...' (top_k={top_k})")
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype='float32')

        distances, indices = self.index.search(query_embedding, k=min(top_k, self.index.ntotal))

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.metadata_store):
                result = self.metadata_store[idx]
                result['score'] = 1 - distances[0][i]  # Distance L2 normalisée -> score pseudo-similarité
                results.append(result)
        
        return results