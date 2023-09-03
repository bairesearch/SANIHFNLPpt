"""SANIHFNLPpy_LayeredSANINode.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
SANIHFNLP Layered SANI Node Classes

"""


import numpy as np
import random

from SANIHFNLPpy_LayeredSANIGlobalDefs import *

def layeredSANINodePropertiesInitialisation(node):
	node.SANIlayerNeuronID = 0
	node.SANIlayer = 0
	node.SANIactivationState = False
	node.activationStatePartial = False	#highlightPartialActivations only (incomplete)
	#node.SANIsequentialInputActiveList = []	#records if sequential inputs are active
	node.SANIinputNodeList = []	#sequential input nodes (lower layer)
	node.SANIoutputNodeList = []	#output nodes (higher layer)
	
class SANIlayer:
	def __init__(self, layerIndex):
		self.layerIndex = layerIndex
		self.sentenceSANINodeList = []
		self.networkSANINodeList = []
		
