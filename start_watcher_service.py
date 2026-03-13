# start_watcher_service.py
# Lance le service de surveillance en mode standalone
# Usage: python start_watcher_service.py

import sys
import signal
import time

from micheline.intel.watchers import WatcherService


def main():
    service = WatcherService()
    
    # Gestion propre du Ctrl+C
    def signal_handler(sig, frame):
        print("\n[Watcher] Interruption détectée (Ctrl+C)...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("  🔍 MICHELINE WATCHER SERVICE")
    print("=" * 60)
    print("\nSurveillance continue des sources du registry.")
    print("Appuyez sur Ctrl+C pour arrêter.\n")
    
    # Seed du registry si vide
    from micheline.intel.entity_registry import EntityRegistry
    registry = EntityRegistry()
    
    if not registry.list_all_active_sources():
        print("[Watcher] ⚠ Aucune source dans le registry.")
        print("[Watcher] Lancement du seed des entités par défaut...")
        
        from micheline.intel.entity_registry import seed_default_entities
        seed_default_entities()
        
        print("[Watcher] ✅ Registry initialisé.")
    
    # Démarrage du service
    service.start(daemon=False)
    
    # Boucle pour garder le programme actif
    try:
        while service._running:
            # Affiche des stats toutes les 60 secondes
            time.sleep(60)
            status = service.get_status()
            print(f"\n[Status] Events: {status['total_events']} | "
                  f"Non traités: {status['unprocessed_events']} | "
                  f"Dernière heure: {status['events_last_hour']}")
    except KeyboardInterrupt:
        pass
    
    service.stop()
    print("\n[Watcher] Service arrêté proprement.")


if __name__ == "__main__":
    main()