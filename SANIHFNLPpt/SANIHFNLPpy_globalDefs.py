"""SANIHFNLPpy_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANIHFNLPpy_main.py

# Usage:
see SANIHFNLPpy_main.py

# Description:
SANIHFNLP - global defs

"""

printVerbose = True

#select SANIHFNLP algorithm options;
from HFNLPpy_globalDefs import useAlgorithmLayeredSANIbiologicalSimulation	#mandatory
from HFNLPpy_globalDefs import useAlgorithmDendriticSANIbiologicalSimulation	#optional	#simulate sequential activation of dendritic input across SANI nodes
	
drawHopfieldGraph = False	#typically use drawBiologicalSimulation only
	
if(drawHopfieldGraph):
	drawHopfieldGraphPlot = True
	drawHopfieldGraphSave = False
	drawHopfieldGraphSentence = False
	drawHopfieldGraphNetwork = True	#default: True	#draw graph for entire network (not just sentence)

def printe(str):
	print(str)
	exit()
