"""SANIHFNLPpy_LayeredSANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SANIHFNLPpy_main.py

# Usage:
see SANIHFNLPpy_main.py

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

def updateLayeredSANIgraph(networkConceptNodeDict, SANIlayerList, sentenceIndex):
	for layerIndex, layer in enumerate(SANIlayerList):
		if(layerIndex < SANInumberOfLayersMax):	#ensure that higher layer SANI nodes can still be added
			#print("layerIndex = ", layerIndex)
			if(layerIndex < SANInumberOfLayersMax-1):
				if(vectoriseComputation):
					updateLayeredSANIlayerVectorised(networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True)
				else:
					updateLayeredSANIlayerStandard(networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True)
			else:
				layerBufferSANINodeListSorted = sortSANInodeList(layer.sentenceSANINodeList)
				if(enableBasicNextWordCausalPredictions):
					recordNextWordCausalPredictions(layerBufferSANINodeListSorted)
			
	if(drawBiologicalSimulation):
		SANIHFNLPpy_LayeredSANIDraw.drawLayeredSANIStatic(SANIlayerList, sentenceIndex)

	clearNetworkActivations(SANIlayerList)
	
	return layerBufferSANINodeListSorted

def sortSANInodeList(sentenceSANINodeList):
	layerBufferSANINodeListSorted = sentenceSANINodeList
	layerBufferSANINodeListSorted.sort(key=lambda x: x.wMin)
	return layerBufferSANINodeListSorted
	
def recordNextWordCausalPredictions(layerBufferSANINodeListSorted):
	SANINeuronPrev = None
	for x1, SANINeuron in enumerate(layerBufferSANINodeListSorted):
		if(x1 > 0):
			if(not SANINeuron.noncontinguousInputNodeAboveCentralContentsStart):
				if(not connectionExists(SANINeuronPrev, SANINeuron, False)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuronPrev, SANINeuron, contextConnection=False, useAlgorithmLayeredSANI=True)
				connection = SANINeuronPrev.HFcausalTargetConnectionLayeredDict[SANINeuron.SANIlayerNeuronID]
				connection.weight += 1
			if(SANINeuron.noncontinguousInputNodeAboveCentralContentsEnd):
				if(not connectionExists(SANINeuron, SANINeuron.noncontinguousInputNodeAbove, False)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuron, SANINeuron.noncontinguousInputNodeAbove, contextConnection=False, useAlgorithmLayeredSANI=True)
				connection = SANINeuron.HFcausalTargetConnectionLayeredDict[SANINeuron.noncontinguousInputNodeAbove.SANIlayerNeuronID]
				connection.SANIoptionalCausalConnection = True
				connection.weight += 1
		SANINeuronPrev = SANINeuron
	
def updateLayeredSANIlayerStandard(networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True):
	layerBufferSANINodeList = layer.sentenceSANINodeList
	#print("len(layerBufferSANINodeList) = ", len(layerBufferSANINodeList))
	sentenceSANINodeList = []
	networkSANINodeList = SANIlayerList[layerIndex+1].networkSANINodeList
	layerNetworkIndex = len(networkSANINodeList)
	networkIndex = len(networkConceptNodeDict)
	for x1, SANINeuron1 in enumerate(layerBufferSANINodeList):
		if(selectActivatedTop):
			layerSANINodeListAssociated = []
		for x2, SANINeuron2 in enumerate(layerBufferSANINodeList):
			if(checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList)):
				if(not connectionExists(SANINeuron1, SANINeuron2, True)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuron1, SANINeuron2, contextConnection=True, useAlgorithmLayeredSANI=True)
				connection = SANINeuron1.HFcontextTargetConnectionLayeredDict[SANINeuron2.SANIlayerNeuronID]
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
			if(checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList)):
				if(checkAssociationTopK(SANINeuron2, layerSANINodeListAssociatedTopK)):
					connection = SANINeuron1.HFcontextTargetConnectionLayeredDict[SANINeuron2.SANIlayerNeuronID]
					if(connection.weight >= SANInodeGenerationHFassociationThreshold):
						addNodeToNextLayer = True
						foundAssociation = True
						if(not connection.SANInodeAssigned):
							if(generateSANINetwork):
								connection.SANInodeAssigned = True
								nodeName = generateSANInodeName(SANINeuron1, SANINeuron2, layerIndex+1, networkIndex)
								if(printVerbose):
									print("updateLayeredSANIlayerStandard: create new conceptNode; ", nodeName)
								nodeGraphType = graphNodeTypeSANIhidden
								connection.SANInode = HopfieldNode(networkIndex, nodeName, nodeGraphType)
								assignWindex(connection.SANInode, SANINeuron1, SANINeuron2)
								connection.SANInode.SANIlayerIndex = layerIndex+1
								connection.SANInode.SANIlayerNeuronID = layerNetworkIndex
								if(checkNodeContiguity(SANINeuron1, SANINeuron2, False)):  
									connection.SANIcontiguousInput = True
								networkSANINodeList.append(connection.SANInode)
								layerNetworkIndex += 1
								networkIndex = addNodeToGraph(connection.SANInode, networkConceptNodeDict, networkIndex)
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

def getSANIlayerNeuronID(nodeName, networkIndex):
	SANIlayerNeuronID = nodeName	#orig: networkIndex (not compatible with generalised HF..ConnectionDict code)
	return SANIlayerNeuronID
	
def assignWindex(SANInode, SANINeuron1, SANINeuron2):
	wApprox = ((SANINeuron1.w + SANINeuron2.w)/2)
	SANInode.w=wApprox
	SANInode.wMin=SANINeuron1.w
	SANInode.wMax=SANINeuron2.w
								
def checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList):
	result = False
	if(checkNonReplicateAssociation(x1, x2)):
		if(checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex)):
			if(checkNodeContiguity(SANINeuron1, SANINeuron2)):
				if(checkNonContiguousAssociationContents(SANINeuron1, SANINeuron2, layerBufferSANINodeList)):
					result = True
	return result

def checkNonReplicateAssociation(x1, x2):
	result = False
	if(x1 != x2 and x2 > x1):	#x2 > x1; ensures that replicate SANI nodes are not added to network (with input swapped)
		result = True	
	return result
	
def checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex):
	result = False
	if(enableSkipLayerConnectivity):
		if((SANINeuron1.SANIlayerIndex == layerIndex) or (SANINeuron2.SANIlayerIndex == layerIndex)):	#at least one node is on the current layer
			result = True
	else:
		result = True
	return result

def checkNodeContiguity(SANINeuron1, SANINeuron2, supportPermutationInvariance=HFassociationPermutationInvariance):
	wContiguityChecks = False
	if(supportPermutationInvariance):
		wContiguityChecks = True
	else:	#wContiguityEnforced
		if(SANINeuron1.wMax+1 == SANINeuron2.wMin):
			wContiguityChecks = True
	return wContiguityChecks	

def checkNonContiguousAssociationContents(SANINeuron1, SANINeuron2, layerBufferSANINodeList):
	result = True
	if(enableNextWordCausalPredictionsPermutationInvariance):
		for node in layerBufferSANINodeList:
			if(not node.SANIcontiguousInput):
				candidateInNodeWindexWindow1 = candidateInNodeWindexCentralWindow(SANINeuron1, node)
				candidateInNodeWindexWindow2 = candidateInNodeWindexCentralWindow(SANINeuron2, node)
				if(candidateInNodeWindexWindow1 or candidateInNodeWindexWindow2):
					if(candidateInNodeWindexWindow1 and candidateInNodeWindexWindow2):
						if(SANINeuron1.wMin == node.wMin+1):	#CHECKTHIS; +1
							SANINeuron1.noncontinguousInputNodeAboveCentralContentsStart = True
						if(SANINeuron2.wMax == node.wMax-1):	#CHECKTHIS; -1
							SANINeuron1.noncontinguousInputNodeAboveCentralContentsEnd = True
							SANINeuron1.noncontinguousInputNodeAbove = node
					else:
						result = False
	return result

def candidateInNodeWindexCentralWindow(candidateNode, nonContiguousNode):
	result = False
	if((candidateNode.wMin > nonContiguousNode.wMin) and (candidateNode.wMax < nonContiguousNode.wMax)):	#CHECKTHIS; >= <=
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
	
def generateSANInodeName(SANINeuron1, SANINeuron2, layerIndex, layerNetworkIndex):
	#nodeName = "l" + str(layerIndex) + "n" + str(layerNetworkIndex)
	nodeName = SANINeuron1.nodeName + "_" + SANINeuron2.nodeName
	return nodeName

def calculateHFproximity(layerBufferSANINodeList, SANINeuron1, SANINeuron2):
	HFproximity = ((len(layerBufferSANINodeList) - min(len(layerBufferSANINodeList), abs(SANINeuron1.w - SANINeuron2.w))) / len(layerBufferSANINodeList)) * HFassociationStrengthProximityBiasLevel
	return HFproximity
	
def clearNetworkActivations(SANIlayerList):
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.networkSANINodeList):
			#print("layerIndex = ", layerIndex)
			SANINode.SANIactivationState = False
			for connectionKey, connection in SANINode.HFcontextTargetConnectionLayeredDict.items():
				if(HFassociationStrengthAtrophy):
					connection.weight += HFassociationStrengthAtrophy
				connection.SANIactivationState = False
				

def addNodeToGraph(conceptNode, networkConceptNodeDict, networkSize):
	if(conceptNode.nodeName not in networkConceptNodeDict):
		#print("addNodeToGraph: conceptNode.nodeName = ", conceptNode.nodeName)
		networkConceptNodeDict[conceptNode.nodeName] = conceptNode
		networkSize += 1
	else:
		print("addNodeToGraph error: conceptNode.nodeName already in networkConceptNodeDict")
		exit()
	return networkSize
		
def connectionExists(nodeSource, nodeTarget, contextConnection):
	result = False
	if(contextConnection):
		if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcontextTargetConnectionLayeredDict):
			result = True
	else:
		if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcausalTargetConnectionLayeredDict):
			result = True
	return result
		
