from math import sqrt, pi

import cv2
import numpy as np
from scipy import stats


def importandgrayscale(path,numFrames,dscale,vidNum):
    #sets up variables to open and save video
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if path is not 0:
        actualNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        actualNumFrames = np.inf
    #set amount of frames want to look at
    counter = 0
    counter1 = 0
    start = numFrames * vidNum
    end = numFrames * (vidNum+ 1)
    if end > actualNumFrames:
        return None
    completeVid = np.zeros((int(height/dscale), int(width/dscale), numFrames), dtype=np.uint8)
    #main body
    while(cap.isOpened() & (counter < end)):
        ret, frame = cap.read()
        #checks to makes sure frame is valid
        if((counter >= start)&(counter < end)):
            if ret == True:
                if frame is not None:
                    frame = cv2.resize(frame,(int(width/dscale),int(height/dscale)),interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    completeVid[:, :, counter1] = gray
                    counter1 += 1
                else:
                    break
            else:
                break
        counter += 1

    cap.release()
    print("finshed Black and white conversion")
    return completeVid


def getDirectModeFrame(completeVid):
    # get mode frame of video
    modeFrame = stats.mode(completeVid, 2)
    modeFrameFinal = (modeFrame[0])[:,:,0]
    print("got direct mode frame")
    return modeFrameFinal

def getDensitytModeFrame(completeVid):
    #set up to variables
    width = completeVid.shape[1]
    height = completeVid.shape[0]
    numFrames = completeVid.shape[2]

    # get mode by using gaussian kernel KDE
    M = np.zeros((height, width, 256))
    stdd = np.std(completeVid,2)
    stdd = stdd*3.5/float(numFrames**(1/3))
    for i in range(256):
        temp = np.zeros((height,width))
        for j in range(numFrames):
            placeholder = i - completeVid[:,:,j]
            temp = temp + (1/(sqrt(2 * pi) * stdd) * np.exp(-0.5 * (placeholder[:,:] / stdd) ** 2))
        M[:,:,i] = temp
        print(i)

    mode = np.argmax(M,2);
    print("got density mode frame")
    return mode


def findmin(completeVid):
    minframe = completeVid.min(2) #return the smalest value for uint8 structure
    return minframe

def createResidual(completeVid,modeImg):
    #sets up variables
    minFrame = findmin(completeVid)
    numFrames = completeVid.shape[2]
    counter = 0
    frame_write = np.zeros(completeVid.shape,dtype=np.float32)
    while (counter < numFrames):
        #subracts mode frame from actual frame
        frame = completeVid[:,:,counter]
        frame_write[:,:,counter]= frame.astype(np.float32) - modeImg.astype(np.float32) + 127 # minFrame.astype(np.float32)
        counter += 1
    #releases video containers
    print("got residual video")
    return frame_write


