#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:45:59 2019

@author: score
"""

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
vidNum = 0 # this change 0 1 2 3 4 

preprocess = ct.preprocessVideo(vidpath,numFrames,dFactor,densityMode,vidNum)

if preprocess is None:
    #return None,None
    

helperFunc.saveVid(preprocess, outputFolder + vidpath[33:-4] + "residual_most_recent.avi")

features, isWater = pd.getFeatures(preprocess,maskpath,dFactor,boxSize,patchSize,numFramesAvg)

#return features, isWater
