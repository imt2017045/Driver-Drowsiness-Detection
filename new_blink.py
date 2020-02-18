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
from Training import Network

import numpy as np
import cv2

import tensorflow as tf

FRAME_MARGIN_BTW_2BLINKS=3
MIN_AMPLITUDE=0.04
MOUTH_AR_THRESH=0.35
MOUTH_AR_THRESH_ALERT=0.30
MOUTH_AR_CONSEC_FRAMES=20

EPSILON=0.01  # for discrete derivative (avoiding zero derivative)

class Blink():
        def __init__(self):

            self.start=0 #frame
            self.startEAR=1
            self.peak=0  #frame
            self.peakEAR = 1
            self.end=0   #frame
            self.endEAR=0
            self.amplitude=(self.startEAR+self.endEAR-2*self.peakEAR)/2
            self.duration = self.end-self.start+1
            self.EAR_of_FOI=0 #FrameOfInterest
            self.values=[]
            self.velocity=0  #Eye-closing velocity

def blink_detector(frame, gray, number_of_frames, First_frame, MCOUNTER, EAR_series):
    rects = detector(gray, 0)
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    loaded_svm = pickle.load(open('Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    if (np.size(rects) != 0):
        number_of_frames = number_of_frames + 1  # we only consider frames that face is detected
        First_frame = False
        old_gray = gray.copy()
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        Mouth = shape[mStart:mEnd]
        MAR = mouth_aspect_ratio(Mouth)
        MouthHull = cv2.convexHull(Mouth)
        cv2.drawContours(frame, [MouthHull], -1, (255, 0, 0), 1)
        if MAR > MOUTH_AR_THRESH:
           MCOUNTER += 1
        elif MAR < MOUTH_AR_THRESH_ALERT:
            if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                MTOTAL += 1
            MCOUNTER = 0
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        EAR_series = shift(EAR_series, -1, cval=ear)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        COUNTER=EMERGENCY(ear,COUNTER)

        if Q.full() and (reference_frame>15):  #to make sure the frame of interest for the EAR vector is int the mid
            EAR_table = EAR_series
            IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
            if Counter4blinks==0:
                Current_Blink = Blink()
            retrieved_blinks, TOTAL_BLINKS, Counter4blinks, BLINK_READY, skip = Blink_Tracker(EAR_series[6],
                                                                                                  IF_Closed_Eyes,
                                                                                                  Counter4blinks,
                                                                                                  TOTAL_BLINKS, skip)
            if (BLINK_READY==True):
                reference_frame=20   #initialize to a random number to avoid overflow in large numbers
                skip = True
                BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames
                for detected_blink in retrieved_blinks:
                    print(detected_blink.amplitude, Last_Blink.amplitude)
                    print(detected_blink.duration, detected_blink.velocity)
                    if(detected_blink.velocity>0):
                        if(cal_flag):
                            u_features.append([BLINK_FRAME_FREQ*100,detected_blink.amplitude,detected_blink.duration,detected_blink.velocity   ])
                        else:
                            ans = solve(BLINK_FRAME_FREQ*100,detected_blink,user_calib)
                            # cv2.putText(frame,"Score : {}".format(ans),(100,10),cv2.FONT_HERSHEY_SIMPLEX,(255,0,255),cv2.LINE_AA)

                Last_Blink.end = -10 # re initialization
            line.set_ydata(EAR_series)
            plot_frame.draw()
            frameMinus7=Q.get()
            cv2.imshow("Frame", frameMinus7)
        elif Q.full():         #just to make way for the new input of the Q when the Q is full
            junk =  Q.get()
        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
            break
    else:
        st=0
        st2=0
        if (First_frame == False):
            leftEye=leftEye.astype(np.float32)
            rightEye = rightEye.astype(np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray,leftEye, None, **lk_params)
            p2, st2, err2 = cv2.calcOpticalFlowPyrLK(old_gray, gray, rightEye, None, **lk_params)
        if np.sum(st)+np.sum(st2)==12 and First_frame==False:
            p1 = np.round(p1).astype(np.int)
            p2 = np.round(p2).astype(np.int)
            leftEAR = eye_aspect_ratio(p1)
            rightEAR = eye_aspect_ratio(p2)
            ear = (leftEAR + rightEAR) / 2.0
            EAR_series = shift(EAR_series, -1, cval=ear)
            #EAR_series[reference_frame] = ear
            leftEyeHull = cv2.convexHull(p1)
            rightEyeHull = cv2.convexHull(p2)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            old_gray = gray.copy()
            leftEye = p1
            rightEye = p2
            COUNTER = EMERGENCY(ear, COUNTER)
        if Q.full() and (reference_frame>15):
            EAR_table = EAR_series
            IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
            if Counter4blinks==0:
                Current_Blink = Blink()
                retrieved_blinks, TOTAL_BLINKS, Counter4blinks, BLINK_READY, skip = Blink_Tracker(EAR_series[6],IF_Closed_Eyes,Counter4blinks,TOTAL_BLINKS, skip)
            if (BLINK_READY==True):
                reference_frame=20   #initialize to a random number to avoid overflow in large numbers
                skip = True
                BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames
                for detected_blink in retrieved_blinks:
                    if(cal_flag):
                        u_features.append([BLINK_FRAME_FREQ*100, detected_blink.amplitude,detected_blink.duration,detected_blink.velocity])
                    else:
                        ans = solve(BLINK_FRAME_FREQ*100,detected_blink,user_calib)
                        #cv2.putText(frame,"Score : {}".format(ans),(100,10),cv2.FONT_HERSHEY_SIMPLEX,(255,0,255),cv2.LINE_AA)
                    with open(output_file, 'ab') as f_handle:
                        f_handle.write(b'\n')
                        np.savetxt(f_handle,[TOTAL_BLINKS,BLINK_FRAME_FREQ*100,detected_blink.amplitude,detected_blink.duration,detected_blink.velocity], delimiter=', ', newline=' ',fmt='%.4f')

                Last_Blink.end = -10 # re initialization
            line.set_ydata(EAR_series)
            plot_frame.draw()
            frameMinus7=Q.get()
            cv2.imshow("Frame", frameMinus7)
        elif Q.full():
            junk = Q.get()
        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
             break
