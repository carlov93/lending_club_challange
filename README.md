# Project Setupt
## Virtual Environment
Create a virtual environment and install packages listed in requirements.txt

## Incorporated own modules to virtual environment
It is important that a python file named "sitecustomize.py" is created with the following content. 
The file must be customized in terms of paths and then added to the folders containing the external libraries. 


import os
import sys

sys.path.append("<Pfad>/ginkgo_analytics/src/ai/models/")
sys.path.append("<Pfad>/ginkgo_analytics/src/utils/")