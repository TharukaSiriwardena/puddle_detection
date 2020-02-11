# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Imagetransformations
import containerFunctions as ct
import predictFunctions as pd
import helperFunc
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
numVids= 5

preprocess = ct.preprocessVideo(vidpath,numFrames,dFactor,densityMode,numVids)

#if preprocess is None:
#    return None,None


helperFunc.saveVid(preprocess, outputFolder + vidpath[-12:-4] + "residual_most_recent.avi")


features, isWater = getFeatures(preprocess,maskpath,dFactor,boxSize,patchSize,numFramesAvg)
#return features, isWater