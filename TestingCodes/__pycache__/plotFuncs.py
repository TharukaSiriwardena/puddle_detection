B
    0��Y�  �               @   s0   d dl Zd dlmZ dd� Zdd� Zdd� ZdS )�    Nc       
      C   s�   | t �||� }| t �t �||�� }t�� \}}|jddid� |�d� t �|d�}t �|d�}d}	|jt �	dd�||	dd	d
� |jt �	dd�|	 ||	ddd
� tjdd� t�
�  d S )N�size�
   )�propzWater vs not Waterr   gffffff�?�   �bZWater)�color�label�rz	Not Waterzupper right)�loc)�np�reshape�invert�plt�subplots�legend�	set_title�average�bar�arange�show)
Z
histMatrix�isWater�
numSamples�water�notwater�fig�axZwater1Z	notwater1Zbarwidth� r   �`/home/score/Documents/Tharuka/water-detection-master/detectPuddleTrain/TestingCodes/plotFuncs.py�plotSpatialFeatures   s    
 r   c             C   s�   |||dd �f }| ||dd �f }|||dd �f }| ||dd �f }	t t|d ��}
t�d� t�d� t�|
|d|
|d� t�d� t�|
|d|
|	d� t��  d S )N�   ��   r   r	   ��   )�range�lenr   �figure�subplot�plotr   )Z
fourierVidZtimeVidZxwaterZywaterZ	xnotwaterZ	ynotwaterZ
waterTime1ZwaterFourier1ZnotwaterTime1ZnotwaterFourier1�Xr   r   r   �plotTimeandFourier   s    


r(   c             C   s  t �� \}}| t�||� }| t�t�||�� }|d d �df }t�|t�|j��}|d d �df }t�|t�|j��}|d d �df }	t�|	t�|	j��}	|d d �df }
t�|
t�|
j��}
|�d� |j||ddd� |j|	|
ddd� t j	d	d
� t �
�  d S )Nr   �   zWater vs not Water�bor   )r   �roz	not waterzupper right)r
   )r   r   r   r   r   �product�shaper   r&   r   r   )Ztemporalsignalsr   r   r   r   r   r   ZwaterxZwateryZ	notwaterxZ	notwateryr   r   r   �PlotTemporalFeatures'   s     
r.   )�numpyr   �matplotlib.pyplot�pyplotr   r   r(   r.   r   r   r   r   �<module>   s   