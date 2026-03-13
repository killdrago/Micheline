# mql5_communicator.py - Communication fichier avec l'EA MQL5
# - Centralisé (chemins + délai de polling)

import os
import time
import config

def wait_for_request():
    print("IA en attente d'une requête de l'EA...")
    poll = float(getattr(config, "MQL5_POLL_INTERVAL_SEC", 0.1))
    while not os.path.exists(config.FLAG_FILE):
        time.sleep(poll)
    print("Requête détectée !")
    return True

def read_request():
    """Lit les données brutes envoyées par l'EA."""
    try:
        with open(config.REQUEST_FILE, 'r', encoding="utf-8", errors="replace") as f:
            data = f.read().strip()
        return data
    except Exception as e:
        print(f"Erreur en lisant la requête : {e}")
        return None

def write_response(response):
    try:
        response_str = f"{response['signal']};{response['lot_size']};{response['take_profit']};{response['stop_loss']}"
        with open(config.RESPONSE_FILE, 'w', encoding="utf-8", errors="replace") as f:
            f.write(response_str)
        print(f"Réponse écrite : {response_str}")
    except Exception as e:
        print(f"Erreur en écrivant la réponse : {e}")
    finally:
        if os.path.exists(config.FLAG_FILE):
            try:
                os.remove(config.FLAG_FILE)
            except Exception:
                pass