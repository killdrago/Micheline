# micheline/rag/ingest.py
# Script exécutable pour indexer une nouvelle source de connaissances.
# S'utilise via le worker: python -m micheline.rag.ingest --source "https://..."

import sys
import argparse
from micheline.rag.document_loader import load_source, split_documents
from micheline.rag.vector_store import KnowledgeBase
import config


def main():
    parser = argparse.ArgumentParser(description="Ingère et indexe une source de données dans la base de connaissances.")
    parser.add_argument("--source", required=True, help="Chemin vers la source (URL, fichier local, ou dossier).")
    parser.add_argument("--chunk-size", type=int, default=config.RAG_CHUNK_SIZE, help="Taille des chunks.")
    parser.add_argument("--chunk-overlap", type=int, default=config.RAG_CHUNK_OVERLAP, help="Chevauchement des chunks.")
    
    args = parser.parse_args()

    print(f"--- Démarrage de l'ingestion pour la source: {args.source} ---")
    
    docs = load_source(args.source)
    if not docs:
        print("Aucun document valide n'a pu être chargé. Fin de l'ingestion.")
        sys.exit(1)
        
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        print("Aucun chunk n'a pu être créé. Fin de l'ingestion.")
        sys.exit(1)
    
    try:
        kb = KnowledgeBase()
        kb.add_documents(chunks)
        print("--- Ingestion terminée avec succès ---")
    except Exception as e:
        print(f"[FATAL] Une erreur est survenue lors de l'initialisation ou de l'indexation de la base de connaissances: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()