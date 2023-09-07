"""SANIHFNLPpy_LayeredSANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
SANIHFNLP Layered SANI - simulate training/inference of biological hybrid SANI-hopfield graph/network based on textual input

training;
	activate concept neurons in order of sentence word order
	detect association between concept neurons
	assign a higher level SANI node for highly associated concept neurons

nodeName = neuronID
"""


import numpy as np
import torch as pt

from SANIHFNLPpy_LayeredSANIGlobalDefs import *
import HFNLPpy_hopfieldOperations
import SANIHFNLPpy_LayeredSANIDraw
from HFNLPpy_hopfieldNodeClass import *

if(drawBiologicalSimulation):
	import SANIHFNLPpy_LayeredSANIDraw

def updateLayeredSANIgraph(SANIlayerList, sentenceIndex):
	for layerIndex, layer in enumerate(SANIlayerList):
		if(layerIndex < SANInumberOfLayersMax):	#ensure that higher layer SANI nodes can still be added
			#print("layerIndex = ", layerIndex)
			if(layerIndex < SANInumberOfLayersMax-1):
				if(vectoriseComputation):
					updateLayeredSANIlayerVectorised(SANIlayerList, layerIndex, layer, generateSANINetwork=True)
				else:
					updateLayeredSANIlayerStandard(SANIlayerList, layerIndex, layer, generateSANINetwork=True)
			else:
				if(enableNextWordCausalPredictions):
					recordNextWordCausalPredictions(layer)
				
	if(drawBiologicalSimulation):
		SANIHFNLPpy_LayeredSANIDraw.drawLayeredSANIStatic(SANIlayerList, sentenceIndex)

	clearNetworkActivations(SANIlayerList)

def recordNextWordCausalPredictions(layer):
	layerBufferSANINodeList = layer.sentenceSANINodeList
	layerBufferSANINodeList.sort(key=lambda x: x.w)
	SANINeuronPrev = None
	for x1, SANINeuron in enumerate(layerBufferSANINodeList):
		if(x1>0):
			if(not HFNLPpy_hopfieldOperations.connectionExists(SANINeuronPrev, SANINeuron, False)):
				HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuronPrev, SANINeuron, contextConnection=False)
			connection = SANINeuronPrev.HFcausalTargetConnectionDict[SANINeuron.SANIlayerNeuronID]
			connection.weight += 1
		SANINeuronPrev = SANINeuron

def updateLayeredSANIlayerStandard(SANIlayerList, layerIndex, layer, generateSANINetwork=True):
	layerBufferSANINodeList = layer.sentenceSANINodeList
	#print("len(layerBufferSANINodeList) = ", len(layerBufferSANINodeList))
	sentenceSANINodeList = []
	networkSANINodeList = SANIlayerList[layerIndex+1].networkSANINodeList
	layerNetworkIndex = len(networkSANINodeList)
	for x1, SANINeuron1 in enumerate(layerBufferSANINodeList):
		if(selectActivatedTop):
			layerSANINodeListAssociated = []
		for x2, SANINeuron2 in enumerate(layerBufferSANINodeList):
			if(checkNonReplicateAssociation(x1, x2) and checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex)):
				if(not HFNLPpy_hopfieldOperations.connectionExists(SANINeuron1, SANINeuron2, True)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuron1, SANINeuron2, contextConnection=True)
				connection = SANINeuron1.HFcontextTargetConnectionDict[SANINeuron2.SANIlayerNeuronID]
				if(checkNodeContiguity(SANINeuron1, SANINeuron2)):
					if(HFassociationStrengthProximityBias):
						HFproximity = calculateHFproximity(layerBufferSANINodeList, SANINeuron1, SANINeuron2)
						#print("HFproximity = ", HFproximity)
						connection.weight += HFproximity
					else:
						connection.weight += 1
					if(selectActivatedTop):
						if(connection.weight > SANInodeGenerationHFassociationThreshold):
							layerSANINodeListAssociated.append(connection)
		if(selectActivatedTop):
			layerSANINodeListAssociated.sort(key=lambda x: x.weight)
			selectActivatedTopKChecked = min([selectActivatedTopK, len(layerSANINodeListAssociated)])
			layerSANINodeListAssociated = layerSANINodeListAssociated[0:selectActivatedTopKChecked]
			layerSANINodeListAssociatedTopK = []
			for connection in layerSANINodeListAssociated:
				layerSANINodeListAssociatedTopK.append(connection.nodeTarget)
		else:
			layerSANINodeListAssociatedTopK = None
		foundAssociation = False
		for x2, SANINeuron2 in enumerate(layerBufferSANINodeList):
			if(checkNonReplicateAssociation(x1, x2) and checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex)):
				if(checkNodeContiguity(SANINeuron1, SANINeuron2)):
					if(checkAssociationTopK(SANINeuron2, layerSANINodeListAssociatedTopK)):
						connection = SANINeuron1.HFcontextTargetConnectionDict[SANINeuron2.SANIlayerNeuronID]
						if(connection.weight >= SANInodeGenerationHFassociationThreshold):
							addNodeToNextLayer = True
							foundAssociation = True
							if(not connection.SANInodeAssigned):
								if(generateSANINetwork):
									connection.SANInodeAssigned = True
									layerNetworkIndex, nodeName = generateSANInodeName(layerIndex+1, layerNetworkIndex)
									if(printVerbose):
										print("create new conceptNode; ", nodeName)
									if(HFassociationPermutationInvariance):
										wApprox = abs(SANINeuron1.w - SANINeuron2.w)
									else:
										wApprox = SANINeuron2.w #wContiguityEnforced
									nodeGraphType = graphNodeTypeSANIhidden
									connection.SANInode = HopfieldNode(layerNetworkIndex, nodeName, nodeGraphType, w=wApprox)
									connection.SANInode.SANIlayerIndex = layerIndex+1
									connection.SANInode.SANIlayerNeuronID = layerNetworkIndex
									networkSANINodeList.append(connection.SANInode)
									layerNetworkIndex += 1
								else:
									addToNextLayer = False
							if(addNodeToNextLayer):
								connection.SANIactivationState = True
								connection.SANInode.SANIactivationState = True
								sentenceSANINodeList.append(connection.SANInode)
		if(enableSkipLayerConnectivity):
			if(not foundAssociation):
				sentenceSANINodeList.append(SANINeuron1)	#append unassociated node to buffer so that it can still be referenced in future layers
				
	SANIlayerList[layerIndex+1].sentenceSANINodeList = sentenceSANINodeList

def checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex):
	result = False
	if(enableSkipLayerConnectivity):
		if((SANINeuron1.SANIlayerIndex == layerIndex) or (SANINeuron2.SANIlayerIndex == layerIndex)):	#at least one node is on the current layer
			result = True
	else:
		result = True
	return result
	
def checkNonReplicateAssociation(x1, x2):
	result = False
	if(x1 != x2 and x2 > x1):	#x2 > x1; ensures that replicate SANI nodes are not added to network (with input swapped)
		result = True	
	return result
	
def checkAssociationTopK(SANINeuron2, layerSANINodeListAssociatedTopK):
	associationTopKChecks = False
	if(selectActivatedTop):
		if(SANINeuron2 in layerSANINodeListAssociatedTopK):
			associationTopKChecks = True
	else:
		associationTopKChecks = True
	return associationTopKChecks
	
def checkNodeContiguity(SANINeuron1, SANINeuron2):
	wContiguityChecks = False
	if(HFassociationPermutationInvariance):
		wContiguityChecks = True
	else:	#wContiguityEnforced
		if(SANINeuron1.w+1 == SANINeuron2.w):
			wContiguityChecks = True
	return wContiguityChecks	
			
def generateSANInodeName(layerIndex, layerNetworkIndex):
	nodeName = "l" + str(layerIndex) + "n" + str(layerNetworkIndex)
	return layerNetworkIndex, nodeName

def calculateHFproximity(layerBufferSANINodeList, SANINeuron1, SANINeuron2):
	HFproximity = ((len(layerBufferSANINodeList) - min(len(layerBufferSANINodeList), abs(SANINeuron1.w - SANINeuron2.w))) / len(layerBufferSANINodeList)) * HFassociationStrengthProximityBiasLevel
	return HFproximity
	
def clearNetworkActivations(SANIlayerList):
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.networkSANINodeList):
			#print("layerIndex = ", layerIndex)
			SANINode.SANIactivationState = False
			for connectionKey, connection in SANINode.HFcontextTargetConnectionDict.items():
				if(HFassociationStrengthAtrophy):
					connection.weight += HFassociationStrengthAtrophy
				connection.SANIactivationState = False
				
