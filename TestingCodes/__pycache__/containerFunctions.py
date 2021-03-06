B
    T��]�  �               @   sd   d dl Z d dlZd dlZd dlZd dlZd dlZd dlZdd� Zdd� Z	dd� Z
dd	� Zd
d� ZdS )�    Nc       	      C   s�   | dk	r2t �| �}|d ks"|dkr2t|�t j��}|d ksB|dkrFd}|d krRd}t�| |||�}|d krnd S |dkr�t�|�}nt�|�}t �	d|� | dk	r�|�
�  t�||�}|S )Nr   ������   zmode_Direct.png)�cv2�VideoCapture�int�get�CAP_PROP_FRAME_COUNT�Imagetransformations�importandgrayscale�getDensitytModeFrame�getDirectModeFrame�imwrite�release�createResidual)	�path�	numFrames�dFactor�densityMode�vidNum�cap�gray�	modeFrame�residual� r   �i/home/score/Documents/Tharuka/water-detection-master/detectPuddleTrain/TestingCodes/containerFunctions.py�preprocessVideo	   s&    

r   c          	   C   s�  | j d }| j d }	| j d }
t�d� |dkr@t�|	|f�}n"t�||�}|d d �d d �df }tjd|ftd�}t�||f�}t�|df�}tt	|d d �t	|d ��}t�||d f�}x�t
|�D ]�}t�||| �}t�||	| �}t�t	|d �|
t	|d � �}|||f |d|f< t�| |||||�||d d �f< ||d d �f }t�| |||||�||d d �f< ||d d �f }t�t�||j�|f�}|||d d �f< q�W td� |�t	�}||fS )Nr   r   �   )�dtype�   z'finished computing unified featureSpace)�shape�random�seed�np�zeros�featuresZ
createMask�bool�maxr   �range�	randrangeZfourierTransformZSpatialFeatures�concatenate�reshape�size�print�astype)ZpreprocessedVid�maskpath�dscale�boxSize�TemporalLength�numbofSamples�	patchSize�numFramesAvg�width�heightr   �mask�isWaterZtemporalArrZ
spatialArrZminrandZtotalFeatures�iZrandxZrandyZrandzZtemporalFeatZ	spaceFeatZcombinedFeaturer   r   r   �getFeaturesPoints#   s6    



   
r:   c
          	   C   s2   t | |||d�}
t|
|||||||	�\}}||fS )Nr   )r   r:   )�vidpathr.   r   r   r   r0   ZNumbofFrameSearchr2   r3   r4   Z
preprocessr$   r8   r   r   r   �moduleBM   s    r<   c             C   s�  d }d }d}x�t �| �D ]�}d}x�t �| d | �D ]�}||krBP || d |d d�  d }| | d | }t|||||||||	|
�
\}}|d kr�|}ntj||fdd�}|d kr�|}ntj||fdd�}|d7 }t|� q4W t|� qW d}x�t �|�D ]�}d}x�t �|| �D ]�}||k�r$P || d | }t|d|||||||	|
�
\}}|d k�rb|}ntj||fdd�}|d k�r�|}ntj||fdd�}|d7 }t|� �qW q�W |d }t�|�}||fS )	N�   r   �/������png)�axisr   �   )�os�listdirr<   r"   r)   r,   �	transpose)�pathToVidsFolder�pathToMasksPondFolderZpathToOtherTexturesr   r   r   r0   r1   r2   r3   r4   �totalFeatureSet�
isWateragg�halfamountofVidsinFolderZfolders�counter�vids�nameMask�nameVid�featurer8   r   r   r   �LoopsThroughAllVidsS   sP    



rP   c
             C   s�   d }
d }d}d}x�t �| �D ]�}||kr*P ||d d�  d }| | }t||||||||||	�
\}}|
d krr|}
ntj|
|fdd�}
|d kr�|}ntj||fdd�}|d7 }t|� qW |d }t�|�}|
|fS )Nr   r   r?   r@   )rA   )rC   rD   r<   r"   r)   r,   rE   )rF   rG   r   r   r   r0   r1   r2   r3   r4   rH   rI   rJ   rK   rL   rM   rN   rO   r8   r   r   r   �JustOneFolder�   s*    
rQ   )rC   r    r   �numpyr"   r	   r$   Z	plotFuncsr   r:   r<   rP   rQ   r   r   r   r   �<module>   s   *5