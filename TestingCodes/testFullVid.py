# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Imagetransformations
import containerFunctions as ct
import predictFunctions as pd
from sklearn.externals import joblib


vidpath = "../ForTesting/videos/water/ocean/ocean_001.avi"
maskpath = 0
outputFolder = "../Results"
numFrames = 200
dFactor = 2
densityMode = 0
boxSize = 5
patchSize = 10
numFramesAvg = 100
vidNum = 0 # this change 0 1 2 3 4 



feature, trueMask = pd.moduleC(vidpath,maskpath,outputFolder,numFrames,dFactor,densityMode,boxSize,patchSize,numFramesAvg,vidNum)

#if feature is None:
#    return None,None

width = feature.shape[1]
height = feature.shape[0]
minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))

# load the SVM model
model = joblib.load('tree_currentBest.pkl')

isWaterFound = np.zeros((height,width),  dtype = np.int)
newShape = feature.reshape((feature.shape[0] * feature.shape[1], feature.shape[2]))
prob = model.predict_proba(newShape)[:, 0]
prob = prob.reshape((feature.shape[0], feature.shape[1]))
probabilityMask = prob
isWaterFound[prob<.5] = True
isWaterFound[prob>.5] = False
isWaterFound = isWaterFound.astype(np.uint8)
isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]
if trueMask is not None:
    trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
    trueMask[trueMask == 1] = 255
isWaterFound[isWaterFound == 1] = 255
beforeReg = outputFolder + str(vidNum) + '_before_regularization' + '.png'
cv2.imwrite(beforeReg, isWaterFound)
probabilityMask = probabilityMask[minrand:height-minrand, minrand:width-minrand]
#probabilityMask = (probabilityMask-np.min(probabilityMask))/(np.max(probabilityMask)-np.min(probabilityMask))
for i in range(11):
    isWaterFound = regularizeFrame(isWaterFound,probabilityMask,.2)
#isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)
width = isWaterFound.shape[1]
height = isWaterFound.shape[0]
isWaterFound = isWaterFound[11:height-11, 11:width-11]
if trueMask is not None:
    cv2.imshow("mask created",isWaterFound)
    trueMask = trueMask[11:height - 11, 11:width - 11]
    cv2.imshow("old mask", trueMask)
    cv2.waitKey(0)
    FigureOutNumbers(isWaterFound, trueMask)
cv2.imwrite(outputFolder + str(vidNum) + 'newMask_direct.png', isWaterFound)
completeVid = Imagetransformations.importandgrayscale(vidpath,numFrames,dFactor,vidNum)
maskedImg = maskFrameWithMyMask(completeVid[11:height-11, 11:width-11,int(numFrames/2)],isWaterFound)
cv2.imwrite(outputFolder + str(vidNum) + 'Masked_frame_from_video.png', maskedImg)
return isWaterFound, trueMask
