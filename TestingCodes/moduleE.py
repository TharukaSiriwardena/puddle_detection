#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:47:15 2019

@author: score
"""

import cv2
import numpy as np
import sys
import Imagetransformations
import containerFunctions as ct
import features
import helperFunc
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
numVids= 5

maskArr = None
for i in range(numVids):
    mask,trueMask = pd.testFullVid(vidpath, maskpath, outputFolder, numFrames, dFactor, densityMode, boxSize, patchSize, numFramesAvg,i)
    if mask is None:
        print("didn't have enough frames to run this many times")
        break
    if maskArr is None:
        maskArr = mask
    else:
        maskArr = np.dstack((maskArr,mask))
        
        
        
        
        
        
        
        
        