"""SANIHFNLPpy_hopfieldGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANIHFNLPpy_main.py

# Usage:
see SANIHFNLPpy_main.py

# Description:
SANIHFNLP Hopfield Graph - generate hopfield graph/network based on textual input

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
import HFNLPpy_hopfieldGraph
from SANIHFNLPpy_globalDefs import *

if(useAlgorithmLayeredSANIbiologicalSimulation):
	#import torch as pt
	import SANIHFNLPpy_LayeredSANI
	import SANIHFNLPpy_LayeredSANINode
	from SANIHFNLPpy_LayeredSANIGlobalDefs import SANInumberOfLayersMax
if(useAlgorithmDendriticSANIbiologicalSimulation):
	from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	import HFNLPpy_DendriticSANI
	if(useDependencyParseTree):
		import HFNLPpy_DendriticSANISyntacticalGraph
		
if(drawHopfieldGraph):
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawSentence
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawNetwork


'''
HFconnectionGraph = None
networkConceptNodeDict = {}
networkSize = 0
'''
SANIlayerList = []
for layerIndex in range(SANInumberOfLayersMax):
	SANIlayerList.append(SANIHFNLPpy_LayeredSANINode.SANIlayer(layerIndex))
	
def generateHopfieldGraphNetwork(articles):
	numberOfSentences = len(articles)

	if(seedHFnetworkSubsequence):
		HFNLPpy_hopfieldGraph.verifySeedSentenceIsReplicant(articles, numberOfSentences)

	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences)	
		

def generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = HFNLPpy_hopfieldGraph.tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	#print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences)

def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences):	
	activationTime = calculateActivationTime(sentenceIndex)
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
	#declare Hopfield graph nodes;	
	HFNLPpy_hopfieldGraph.generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList)

	if(useAlgorithmLayeredSANIbiologicalSimulation):
		for conceptNode in sentenceConceptNodeList:	
			if(useAlgorithmLayeredSANIbiologicalSimulation):
				conceptNode.SANIlayerNeuronID = conceptNode.networkIndex
				conceptNode.SANIlayerIndex = 0

	if(useAlgorithmLayeredSANIbiologicalSimulation):
		SANIlayerList[0].networkSANINodeList = list(HFNLPpy_hopfieldGraph.networkConceptNodeDict.values())
		SANIlayerList[0].sentenceSANINodeList = sentenceConceptNodeList
		
	trainSentence = True
	if(sentenceIndex == numberOfSentences-1):
		if(seedHFnetworkSubsequence):
			trainSentence = False
			
	if(trainSentence):
		#create Hopfield graph connections (non-useAlgorithmDendriticSANIbiologicalSimulation);
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			sentenceSANINodeList = SANIHFNLPpy_LayeredSANI.updateLayeredSANIgraph(HFNLPpy_hopfieldGraph.networkConceptNodeDict, SANIlayerList, sentenceIndex)
			HFNLPpy_hopfieldGraph.recalculateHopfieldGraphNetworkSize()
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			HFNLPpy_DendriticSANI.trainBiologicalHFnetwork(HFNLPpy_hopfieldGraph.networkConceptNodeDict, sentenceIndex, sentenceSANINodeList, numberOfSentences)	
	else:
		#predict Hopfield graph flow;
		seedSentenceConceptNodeList = sentenceConceptNodeList
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			HFNLPpy_DendriticSANI.seedBiologicalHFnetwork(HFNLPpy_hopfieldGraph.networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences)			
		else:
			printe("SANIHFNLPpy_hopfieldGraph:generateHopfieldGraphSentence error: !trainSentence requires useAlgorithmDendriticSANIbiologicalSimulation")
			
	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			hopfieldGraphDrawSentence.drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, HFNLPpy_hopfieldGraph.networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave)
		if(drawHopfieldGraphNetwork):
			hopfieldGraphDrawNetwork.drawHopfieldGraphNetworkStatic(sentenceIndex, HFNLPpy_hopfieldGraph.networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave)

	result = True
	return result
