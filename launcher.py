# launcher.py

import sys
import subprocess
import time

# Attend une seconde pour être sûr que le processus principal est bien fermé
time.sleep(1)

# Relance le script main.py dans un nouveau processus indépendant
subprocess.Popen([sys.executable, "main.py"])