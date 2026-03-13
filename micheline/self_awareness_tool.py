# micheline/self_awareness_tool.py
# Outil permettant à l'IA d'inspecter son propre code source.

import os
from typing import Dict, List, Optional
import config

# Définir le répertoire racine du projet pour éviter les ambiguïtés
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Liste des fichiers/dossiers à ignorer pour ne pas polluer le contexte
IGNORE_PATTERNS = [
    '__pycache__', '.git', '.idea', '.vscode', 'venv', 'attachments_tmp',
    '.DS_Store', 'trained_models', '.onnx', '.joblib', '.keras', '.gguf',
    'knowledge.faiss', 'knowledge.meta.json', 'assistant.sqlite'
]

# Description "manuelle" du rôle des fichiers clés pour aider l'IA
FILE_SUMMARIES = {
    "main.py": "Coeur de l'application. Gère l'interface graphique (Tkinter), la logique des conversations, les interactions utilisateur et orchestre les appels aux autres modules.",
    "config.py": "Fichier de configuration central. Contient tous les paramètres ajustables de l'IA, du trading, du RAG et de l'UI. C'est ici que se trouvent les seuils et les stratégies.",
    "worker.py": "Processus d'arrière-plan. Exécute les tâches longues (entraînement, backtest, ingestion) de manière asynchrone pour ne pas bloquer l'interface.",
    "trainer.py": "Module d'entraînement des modèles de prédiction. Gère la création des features, la préparation des données et l'entraînement des modèles Keras.",
    "model_manager.py": "Définit l'architecture des réseaux de neurones (EnsembleAIBrain) utilisés pour la prédiction des signaux de trading.",
    "trade_analyzer.py": "Script de backtest détaillé. Simule les trades basés sur les prédictions du modèle et génère un rapport de performance complet.",
    "sl_tp_optimizer.py": "Optimiseur des niveaux de Stop Loss et Take Profit en utilisant des simulations rapides pour trouver les multiplicateurs d'ATR les plus rentables.",
    "feature_optimizer.py": "Optimiseur d'indicateurs techniques. Teste différentes combinaisons de features pour trouver le set le plus performant pour chaque paire.",
    "ai_bot.py": "Le bot de trading en temps réel. S'intègre avec MQL5, utilise les modèles entraînés pour générer des prédictions et prendre des décisions de trading.",
    "micheline/rag/vector_store.py": "Gère la mémoire de connaissances (base de données vectorielle FAISS). Permet d'indexer et de rechercher des informations.",
    "micheline/self_awareness_tool.py": "Ce fichier même. Fournit à l'IA la capacité de scanner et de comprendre sa propre structure de code."
}

def is_ignored(path: str) -> bool:
    """Vérifie si un chemin doit être ignoré."""
    path_parts = path.split(os.sep)
    for part in path_parts:
        for pattern in IGNORE_PATTERNS:
            if part.endswith(pattern):
                return True
    return False

def get_project_structure() -> str:
    """Scanne le projet et retourne une arborescence textuelle des fichiers pertinents."""
    structure = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filtrer les répertoires à ignorer
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
        
        level = root.replace(PROJECT_ROOT, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # S'assure que le dossier racine est bien affiché
        if root == PROJECT_ROOT:
            structure.append(f"{os.path.basename(PROJECT_ROOT)}/")
        else:
            structure.append(f'{indent}📂 {os.path.basename(root)}/')
        
        sub_indent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            file_path = os.path.join(root, f)
            if not is_ignored(file_path):
                structure.append(f'{sub_indent}📄 {f}')
                
    return "\n".join(structure)

def get_file_content(relative_path: str) -> Optional[str]:
    """Lit le contenu d'un fichier du projet à partir de son chemin relatif."""
    if ".." in relative_path: # Simple mesure de sécurité
        return "Erreur : Le chemin ne doit pas contenir '..'."
        
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.exists(full_path) or is_ignored(full_path) or not os.path.isfile(full_path):
        return None
        
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Erreur de lecture du fichier : {e}"

def generate_self_awareness_context(focus_files: Optional[List[str]] = None) -> str:
    """
    Génère un contexte complet pour que l'IA comprenne sa propre structure et son code.
    'focus_files' est une liste de chemins relatifs (ex: ['config.py', 'trade_analyzer.py'])
    sur lesquels l'IA doit se concentrer.
    """
    context = []
    
    # 1. Introduction
    context.append("--- CONTEXTE D'AUTO-ANALYSE DE L'IA MICHELINE ---")
    context.append("Tu es Micheline. Le contexte suivant décrit ta propre structure de code et ton fonctionnement interne. Utilise ces informations pour analyser tes performances et suggérer des améliorations.")
    
    # 2. Arborescence du projet
    context.append("\n--- ARBORESCENCE DU PROJET ---")
    context.append(get_project_structure())
    
    # 3. Résumé des fichiers clés
    context.append("\n--- RÔLE DES FICHIERS CLÉS ---")
    for filename, summary in FILE_SUMMARIES.items():
        context.append(f"- {filename}: {summary}")
        
    # 4. Contenu des fichiers ciblés (si demandé)
    if focus_files:
        context.append("\n--- CONTENU DES FICHIERS CIBLÉS POUR L'ANALYSE ---")
        for file_path in focus_files:
            content = get_file_content(file_path)
            if content:
                context.append(f"\n--- Début du fichier : {file_path} ---")
                context.append(content)
                context.append(f"--- Fin du fichier : {file_path} ---")
            else:
                context.append(f"\n--- Fichier non trouvé ou ignoré : {file_path} ---")
                
    return "\n".join(context)