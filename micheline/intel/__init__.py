# micheline/intel/__init__.py
"""
Module Intel.

Important:
- On n'importe PAS automatiquement watchers ici.
  Ça évite les erreurs d'import (RawEventsDB manquant, circular imports, etc.).
- Pour utiliser le watcher:
    from micheline.intel.watchers import WatcherService
"""

from .entity_registry import EntityRegistry, seed_default_entities

__all__ = [
    "EntityRegistry",
    "seed_default_entities",
]
