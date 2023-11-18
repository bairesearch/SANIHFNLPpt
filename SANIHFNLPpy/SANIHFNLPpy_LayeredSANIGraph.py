"""SANIHFNLPpy_LayeredSANIGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANIHFNLPpy_main.py

# Usage:
see SANIHFNLPpy_main.py

# Description:
SANIHFNLP Layered SANI Graph - generate layered SANI graph/network based on textual input

"""

import SANIHFNLPpy_LayeredSANI
import SANIHFNLPpy_LayeredSANINode
from SANIHFNLPpy_LayeredSANIGlobalDefs import SANInumberOfLayersMax

SANIlayerList = []
for layerIndex in range(SANInumberOfLayersMax):
	SANIlayerList.append(SANIHFNLPpy_LayeredSANINode.SANIlayer(layerIndex))

def generateLayeredSANIGraphSentence(HFconnectionGraphObject, sentenceIndex, tokenisedSentence, sentenceConceptNodeList, networkConceptNodeDict):	
	
	for conceptNode in sentenceConceptNodeList:	
		conceptNode.SANIlayerNeuronID = conceptNode.networkIndex
		conceptNode.SANIlayerIndex = 0

	SANIlayerList[0].networkSANINodeList = list(networkConceptNodeDict.values())
	SANIlayerList[0].sentenceSANINodeList = sentenceConceptNodeList
		
	sentenceSANINodeList = SANIHFNLPpy_LayeredSANI.updateLayeredSANIgraph(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, sentenceIndex)
	
	return sentenceSANINodeList
