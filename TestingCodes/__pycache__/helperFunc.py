B
    0��Y(  �               @   s$   d dl Z d dlZdd� Zdd� ZdS )�    Nc       
      C   s�   | � tj�} | jd }| jd }d}tjd� }t�|||||fd�}d}| jd }xX||k r�| d d �d d �|f }	|�|	� t�d|	� t�	d�d	@ t
d
�kr�P |d7 }qTW t��  |��  d S )N�   r   �   �XVIDF�   Z
justRunVidi�  ��   �q)�astype�np�uint8�shape�cv2�VideoWriter_fourcc�VideoWriter�write�imshow�waitKey�ord�destroyAllWindows�release)
�arr�
outputName�width�height�fps�fourcc�out�counter�	numFrames�frame� r   �a/home/score/Documents/Tharuka/water-detection-master/detectPuddleTrain/TestingCodes/helperFunc.py�playVid   s"    





r!   c       
      C   s�   | � tj�} | jd }| jd }d}tjd� }t�|||||fd�}d}| jd }x4||k r�| d d �d d �|f }	|�|	� |d7 }qTW t��  |�	�  d S )Nr   r   r   r   Fr   )
r   r	   r
   r   r   r   r   r   r   r   )
r   r   r   r   r   r   r   r   r   r   r   r   r    �saveVid   s    





r"   )r   �numpyr	   r!   r"   r   r   r   r    �<module>   s   