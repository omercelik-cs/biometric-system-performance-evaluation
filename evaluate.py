import numpy as np
import scipy.stats as stats
import pylab as pl # comes with matplotlib
import sys
import math

def determineThresholds(geniune, impostor, thresholdNumber):
	tMax = max(max(geniune),max(impostor))
	tMin = min(min(impostor), min(geniune))
	increaseAmount = (tMax- tMin)/ thresholdNumber
	thresholdArray = []
	i = tMin
	while(i<tMax):
		thresholdArray.append(i)
		i+=increaseAmount

	return thresholdArray

def computeFARGivenThreshold(impostor,t):
	impostorMatching = 0
	for score in impostor:
		if score > t:
			impostorMatching+=1

	return impostorMatching/float(len(impostor))

def computeFRRGivenThreshold(geniune,t):
	geniuneMatching = 0
	for score in geniune:
		if score < t:
			geniuneMatching+=1

	return (geniuneMatching/float(len(geniune))) 

def findIntersection(FARScores, FRRScores):
	i = 1
	leastDifference = abs(FARScores[0] - FRRScores[0])
	leastIndex = 0
	while(i<len(FARScores)):
		temp = (abs(FARScores[i] - FRRScores[i]))
		if (temp< leastDifference):
			leastDifference = temp
			leastIndex = i
		i+=1
	return leastIndex

def findFRRGivenFAR(FARScores, point):
	i = 1
	leastDifference = abs(FARScores[0] - point)
	leastIndex = 0
	while(i<len(FARScores)):
		temp = (abs(FARScores[i] - point))
		if (temp< leastDifference):
			leastDifference = temp
			leastIndex = i
		i+=1
	return leastIndex

def calculateEER(geniune, impostor,thresholdNumber):
	thresholds = determineThresholds(geniune,impostor,thresholdNumber)
	FARScores = []
	GMRScores = []
	FRRScores = []
	for threshold in thresholds:
		FARScores.append(computeFARGivenThreshold(impostor, threshold))
		temp = computeFRRGivenThreshold(geniune, threshold)
		FRRScores.append(temp)
		GMRScores.append(1-temp)
	index = findIntersection(FARScores,FRRScores)
	EER = (FARScores[index] + FRRScores[index]) /2
	EERThreshold = thresholds[index]

	print ("EER: "+ '{:.2%}'.format(EER) + "\tEER Threshold: " + "%.1f" % EERThreshold)
	print( "\nFRR: "+ '{:.2%}'.format(FRRScores[findFRRGivenFAR(FARScores,0.001)]) + "\tat FAR point: 0.1%")
	print( "FRR: "+ '{:.2%}'.format(FRRScores[findFRRGivenFAR(FARScores,0.01)]) + "\tat FAR point: 1%")
	print( "FRR: "+ '{:.2%}'.format(FRRScores[findFRRGivenFAR(FARScores,0.1)]) + "\tat FAR point: 10%")
	return FARScores, GMRScores # these are returned for plots

def plotRocCurve(FARScores,GMRScores):
	fig = pl.figure(2)
	ax = fig.add_subplot(111)
	ax.set_title('ROC Analysis')
	ax.set_xlabel('FAR')
	ax.set_ylabel('GMR')
	pl.plot(FARScores,GMRScores)
	pl.show() 

def plotRocCurveWithLogarithm(FARScores,GMRScores):
	logFAR = []
	for item in FARScores:
		try:
			logFAR.append(math.log10(item))
		except:
			logFAR.append(0)

	fig = pl.figure(3)
	ax = fig.add_subplot(111)
	ax.set_title('Detailed ROC Analysis')
	ax.set_xlabel('log(FAR)')
	ax.set_ylabel('GMR')
	pl.plot(logFAR,GMRScores)
	pl.show() 

def plotDistribution(geniune, impostor):
	fit = stats.norm.pdf(geniune, np.mean(geniune), np.std(geniune))
	fit2 = stats.norm.pdf(impostor, np.mean(impostor), np.std(impostor)) 

	fig = pl.figure(1)
	ax = fig.add_subplot(111)
	ax.set_title('geniune impostor score distribution')
	ax.set_xlabel('score')
	ax.set_ylabel('pdf')
	ax.text(0.02, 0.99, 'geniune',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='blue', fontsize=10)
	ax.text(0.02, 0.95, 'impostor',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='orange', fontsize=10)
	ax.plot(geniune,fit)
	ax.plot(impostor,fit2)
	pl.show() 

def getGeniuneImpostorScores(SM, classLabels):
	geniune = []
	impostor = []
	i =1
	for row in SM:
		label = classLabels[i-1]
		for j in range(i,len(row)):
			if classLabels[j] == label:
				if row[j] != "NaN":
						geniune.append(row[j])
			else:
				if row[j] != "NaN":
					impostor.append(row[j])
		i+=1
	geniune = list(map(float, geniune))
	impostor = list(map(float, impostor))
	return geniune,impostor

def main(data, labels, thresholdNumber=1000):
	# file read stuff and organize data
	with open(data) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]
	dataList = []
	for item in content:
		templist = []
		templist = item.split(",")
		dataList.append(templist)

	with open(labels) as f:
	    labelList = f.readlines()
	labelList = [x.strip() for x in labelList] 

	geniune, impostor = getGeniuneImpostorScores(dataList,labelList)
	FARScores, GMRScores = calculateEER(geniune,impostor,thresholdNumber)
	plotDistribution(sorted(geniune),sorted(impostor))
	plotRocCurve(FARScores, GMRScores)
	plotRocCurveWithLogarithm(FARScores,GMRScores)

if __name__ == "__main__":
	if len(sys.argv) == 3:
		main(sys.argv[1],sys.argv[2])
	elif len(sys.argv) == 4:
		main(sys.argv[1],sys.argv[2],float(sys.argv[3]))
	else:
		print("Wrong number of input argument: either 2 or 3")
