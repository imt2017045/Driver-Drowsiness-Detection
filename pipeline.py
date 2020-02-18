#Reference:https://www.pyimagesearch.com/
#This file  detects blinks, their parameters and analyzes them[the final main code]
# import the necessary packages
from __future__ import print_function

from scipy.spatial import distance as dist
import scipy.ndimage.filters as signal

from imutils import face_utils

import datetime
import imutils
import dlib

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import*
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage.interpolation import shift
import pickle
from queue import Queue

# import the necessary packages

import numpy as np
import cv2

#############
####Main#####
#############
def main():
    user_calib = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    u_features=[]
    input_net=None
    keep_p=None
    final = []
    training =None
    features =[]
    flag_for_window=False
    FOLD_NO = 4

    COUNTER = 0
    MCOUNTER=0
    TOTAL = 0
    MTOTAL=0
    TOTAL_BLINKS=0
    Counter4blinks=0
    skip=False # to make sure a blink is not counted twice in the Blink_Tracker function
    Last_Blink=Blink()

    print("[INFO] starting video stream thread...")
    lk_params=dict( winSize  = (13,13), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    EAR_series=np.zeros([13])
    Frame_series=np.linspace(1,13,13)
    reference_frame=0
    First_frame=True
    top = tk.Tk()
    frame1 = Frame(top)
    frame1.grid(row=0, column=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_frame =FigureCanvasTkAgg(fig, master=frame1)
    plot_frame.get_tk_widget().pack(side=tk.BOTTOM, expand=True)
    plt.ylim([0.0, 0.5])
    line, = ax.plot(Frame_series,EAR_series)
    plot_frame.draw()
    # loop over frames from the video stream
    stream = cv2.VideoCapture(path)
    start = datetime.datetime.now()
    number_of_frames=0
    print("Calibrating user face features ----------------------------------------\n")
    flag_for_calibration = 0 #calibarate for around 2000 frames
    CALIB_LENGTH  = 500
    cal_flag = True
    while True:
        if(len(final)>5):
            print(">>>>>>>>>>>>>>>>>>>>>{}<<<<<<<<<<<<<<<<<<<<<<<<<<".format((sum(final)/len(final)+0.0)))
        if(flag_for_calibration == CALIB_LENGTH):
            cal_flag = False
            calibrate(u_features)
            print("Finished Calibration-----------")
        flag_for_calibration +=1
        (grabbed, frame) = stream.read()
        if not grabbed:
            print('not grabbed')
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Brighten the image(Gamma correction)
        reference_frame = reference_frame + 1
        gray=adjust_gamma(gray,gamma=1.5)
        Q.put(frame)
        end = datetime.datetime.now()
        ElapsedTime=(end - start).total_seconds()
        blink = blink_detector(gray, )
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF:
                 break
    stream.release()
    cv2.destroyAllWindows()
