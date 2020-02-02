# Setup Virtual Environment
Es ist wichtig, dass ein python file mit dem Namen "sitecustomize.py" erstellt wird, der folgenden Inhalt hat. 
Die Datei muss bezüglich der Pfade angepasst werden und dann zu den Ordner mit den externen Libraries hinzugefügt werden. 

import os
import sys

sys.path.append("<Pfad>/ginkgo_analytics/src/ai/models/")
sys.path.append("<Pfad>/ginkgo_analytics/src/utils/")