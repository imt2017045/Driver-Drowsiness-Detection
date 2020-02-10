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

####################################Global Variables###########################

user_calib = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
u_features=[]
input_net=None
keep_p=None
final = []
training =None
features =[]
flag_for_window=False
FOLD_NO = 4
##########################Required set of functions####################################

def initialize_model(output_size,feature_size,batch_size,Pre_fc1_size,Post_fc1_size_per_layer,embb_size, embb_size2,Post_fc2_size,
hstate_size,num_layers,step_size,drop_out_p,lr,th,):  #total_input is the shuffled input with size=[Total data points, T,F]
    global input_net,keep_p,training
    tf.reset_default_graph()
    L2loss=0
    input_net = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='bacth_in')
    labels = tf.placeholder(tf.float32, shape=(None, output_size), name='labels_net')  #size=[batch,1]
    keep_p=tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool,name='phase_train')
    output,end_points,concati=Network(input=input_net,Pre_fc1_size=Pre_fc1_size,Post_fc1_size_per_layer=Post_fc1_size_per_layer,
                   embb_size=embb_size,embb_size2=embb_size2,Post_fc2_size=Post_fc2_size,hstate_size=hstate_size,num_layers=num_layers,
                   feature_size=feature_size,step_size=step_size,output_size=output_size,keep_p=keep_p,training=training)
    error=tf.abs(output-labels)
    loss2 =tf.maximum(0.0,tf.square(error)-th)
    loss2 = tf.reduce_mean(loss2)
    variable_path='./'
    with tf.variable_scope('last_fc',reuse=True):
        last_fc_weights = tf.get_variable('weights')
    with tf.variable_scope('post_fc2',reuse=True):
        post_fc2_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings',reuse=True):
        embeddings_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings2',reuse=True):
        embeddings_weights2 = tf.get_variable('weights')
    with tf.variable_scope('pre_fc1',reuse=True):
        pre_fc1_weights = tf.get_variable('weights')

    with tf.variable_scope('post_fc1',reuse=True):
        for lay in range(num_layers):
            post_fc1_weights = tf.get_variable('weights_%s' % lay)
            L2loss=tf.nn.l2_loss(post_fc1_weights)+L2loss
    #
    loss=loss2+0.1 * (tf.nn.l2_loss(last_fc_weights) +tf.nn.l2_loss(pre_fc1_weights) + L2loss+
                       tf.nn.l2_loss(post_fc2_weights) + tf.nn.l2_loss(embeddings_weights)+ tf.nn.l2_loss(embeddings_weights2))
    optimizer=tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, variable_path+'my_model{}'.format(FOLD_NO))
    return sess,output,end_points
sess,output,end_points = initialize_model(output_size=1,feature_size=4,batch_size=64,Pre_fc1_size=32,Post_fc1_size_per_layer=16,
                embb_size=16,embb_size2=16,Post_fc2_size=8,hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,
                                              lr=0.000053,th=1.253)
def predict_image(inputs):
    global sess, end_points, output
    inputs = np.asarray(inputs)
    inputs = inputs.reshape((1,30,4))
    predicts_Test,mid_vT = sess.run([ output,end_points],feed_dict={input_net: inputs,keep_p:1.0,training:False})
    return predicts_Test


def normalize_blinks(num_blinks, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur, Vel, u_Vel,
                     sigma_Vel):
    normalized_blinks = np.zeros([num_blinks, 4])
    normalized_Freq = (Freq - u_Freq) / sigma_Freq
    normalized_blinks[:, 0] = normalized_Freq
    normalized_Amp = (Amp  - u_Amp) / sigma_Amp
    normalized_blinks[:, 1] = normalized_Amp
    normalized_Dur = (Dur  - u_Dur) / sigma_Dur
    normalized_blinks[:, 2] = normalized_Dur
    normalized_Vel = (Vel - u_Vel) / sigma_Vel
    normalized_blinks[:, 3] = normalized_Vel
    return normalized_blinks

def calibrate(k):
    frq = np.array([k[i][0] for i in range(len(k))])
    amp = np.array([k[i][1] for i in range(len(k))])
    dur = np.array([k[i][2] for i in range(len(k))])
    vel = np.array([k[i][3] for i in range(len(k))])
    user_calib[0][0] = np.mean(frq)
    user_calib[1][0] = np.mean(amp)
    user_calib[2][0] = np.mean(dur)
    user_calib[3][0] = np.mean(vel)
    user_calib[0][1] = np.std(frq)
    user_calib[1][1] = np.std(amp)
    user_calib[2][1] = np.std(dur)
    user_calib[3][1] = np.std(vel)
    normalized = normalize_blinks(len(k),frq,user_calib[0][0],user_calib[0][1],amp,user_calib[1][0],user_calib[1][1],dur,user_calib[2][0],user_calib[2][1],vel,user_calib[3][0],user_calib[3][1])

def solve(freq,blink,calibration):
    global final,features,flag_for_window
    if(flag_for_window):
        features = features[1::]
        blinks = [blink.amplitude,blink.duration ,blink.velocity]
        features.append(normalize_blinks(1,freq,user_calib[0][0],user_calib[0][1],blinks[0],user_calib[1][0],user_calib[1][1],blinks[1],user_calib[2][0],user_calib[2][1],blinks[2],user_calib[3][0],user_calib[3][1]))
        ans = predict_image(features)
        final.append(ans)
        return ans
    elif(len(features)==30):
        flag_for_window = True
        ans  = predict_image(features)
        return ans
    else:
        blinks = [blink.amplitude,blink.duration ,blink.velocity]
        features.append(normalize_blinks(1,freq,user_calib[0][0],user_calib[0][1],blinks[0],user_calib[1][0],user_calib[1][1],blinks[1],user_calib[2][0],user_calib[2][1],blinks[2],user_calib[3][0],user_calib[3][1]))
        # temp_features = 
        print(features)
        return ;

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def blink_detector(output_textfile,input_video):
    Q = Queue(maxsize=7)
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

    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C<0.1:           #practical finetuning due to possible numerical issue as a result of optical flow
            ear=0.3
        else:
            ear = (A + B) / (2.0 * C)

        ear = min(ear,0.45)
        return ear

    def mouth_aspect_ratio(mouth):
        A = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[12], mouth[16])
        if C<0.1:           #practical finetuning
            mar=0.2
        else:
            mar = (A ) / (C)
        return mar

    def EMERGENCY(ear, COUNTER):
        if ear < 0.21:
            COUNTER += 1
            if COUNTER >= 50:
                print('EMERGENCY SITUATION (EYES TOO LONG CLOSED)')
                print(COUNTER)
                COUNTER = 0
        else:
            COUNTER=0
        return COUNTER

    def Linear_Interpolate(start,end,N):
        m=(end-start)/(N+1)
        x=np.linspace(1,N,N)
        y=m*(x-0)+start
        return list(y)

    def Ultimate_Blink_Check():
        retrieved_blinks=[]
        MISSED_BLINKS=False
        values=np.asarray(Last_Blink.values)
        THRESHOLD=0.4*np.min(values)+0.6*np.max(values)   # this is to split extrema in highs and lows
        N=len(values)
        Derivative=values[1:N]-values[0:N-1]    #[-1 1] is used for derivative
        i=np.where(Derivative==0)
        if len(i[0])!=0:
            for k in i[0]:
                if k==0:
                    Derivative[0]=-EPSILON
                else:
                    Derivative[k]=EPSILON*Derivative[k-1]

        M=N-1    #len(Derivative)
        ZeroCrossing=Derivative[1:M]*Derivative[0:M-1]
        x = np.where(ZeroCrossing < 0)
        xtrema_index=x[0]+1
        XtremaEAR=values[xtrema_index]
        Updown=np.ones(len(xtrema_index))        # 1 means high, -1 means low for each extremum
        Updown[XtremaEAR<THRESHOLD]=-1           #this says if the extremum occurs in the upper/lower half of signal
        Updown=np.concatenate(([1],Updown,[1]))
        XtremaEAR=np.concatenate(([values[0]],XtremaEAR,[values[N-1]]))
        xtrema_index = np.concatenate(([0], xtrema_index,[N - 1]))
        Updown_XeroCrossing = Updown[1:len(Updown)] * Updown[0:len(Updown) - 1]
        jump_index = np.where(Updown_XeroCrossing < 0)
        numberOfblinks = int(len(jump_index[0]) / 2)
        selected_EAR_First = XtremaEAR[jump_index[0]]
        selected_EAR_Sec = XtremaEAR[jump_index[0] + 1]
        selected_index_First = xtrema_index[jump_index[0]]
        selected_index_Sec = xtrema_index[jump_index[0] + 1]
        if numberOfblinks>1:
            MISSED_BLINKS=True
        if numberOfblinks ==0:
            print(Updown,Last_Blink.duration)
            print(values)
            print(Derivative)
        for j in range(numberOfblinks):
            detected_blink=Blink()
            detected_blink.start=selected_index_First[2*j]
            detected_blink.peak = selected_index_Sec[2*j]
            detected_blink.end = selected_index_Sec[2*j + 1]

            detected_blink.startEAR=selected_EAR_First[2*j]
            detected_blink.peakEAR = selected_EAR_Sec[2*j]
            detected_blink.endEAR = selected_EAR_Sec[2*j + 1]

            detected_blink.duration=detected_blink.end-detected_blink.start+1
            detected_blink.amplitude=0.5*(detected_blink.startEAR-detected_blink.peakEAR)+0.5*(detected_blink.endEAR-detected_blink.peakEAR)
            detected_blink.velocity=(detected_blink.endEAR-selected_EAR_First[2*j+1])/(detected_blink.end-selected_index_First[2*j+1]+1) #eye opening ave velocity
            retrieved_blinks.append(detected_blink)
        return MISSED_BLINKS,retrieved_blinks

    def Blink_Tracker(EAR,IF_Closed_Eyes,Counter4blinks,TOTAL_BLINKS,skip):
        global user_calib
        BLINK_READY=False
        if int(IF_Closed_Eyes)==1:
            Current_Blink.values.append(EAR)
            Current_Blink.EAR_of_FOI=EAR      #Save to use later
            if Counter4blinks>0:
                skip = False
            if Counter4blinks==0:
                Current_Blink.startEAR=EAR    #EAR_series[6] is the EAR for the frame of interest(the middle one)
                Current_Blink.start=reference_frame-6   #reference-6 points to the frame of interest which will be the 'start' of the blink
            Counter4blinks += 1
            if Current_Blink.peakEAR>=EAR:    #deciding the min point of the EAR signal
                Current_Blink.peakEAR =EAR
                Current_Blink.peak=reference_frame-6
        else:
            if Counter4blinks <2 and skip==False :           # Wait to approve or reject the last blink
                if Last_Blink.duration>15:
                    FRAME_MARGIN_BTW_2BLINKS=8
                else:
                    FRAME_MARGIN_BTW_2BLINKS=1
                if ( (reference_frame-6) - Last_Blink.end) > FRAME_MARGIN_BTW_2BLINKS:
                    if  Last_Blink.peakEAR < Last_Blink.startEAR and Last_Blink.peakEAR < Last_Blink.endEAR and Last_Blink.amplitude>MIN_AMPLITUDE and Last_Blink.start<Last_Blink.peak:
                        if((Last_Blink.startEAR - Last_Blink.peakEAR)> (Last_Blink.endEAR - Last_Blink.peakEAR)*0.25 and (Last_Blink.startEAR - Last_Blink.peakEAR)*0.25< (Last_Blink.endEAR - Last_Blink.peakEAR)): # the amplitude is balanced
                            BLINK_READY = True
                            Last_Blink.values=signal.convolve1d(Last_Blink.values, [1/3.0, 1/3.0,1/3.0],mode='nearest')
                            [MISSED_BLINKS,retrieved_blinks]=Ultimate_Blink_Check()
                            TOTAL_BLINKS =TOTAL_BLINKS+len(retrieved_blinks)  # Finally, approving/counting the previous blink candidate
                            Counter4blinks = 0
                            print("MISSED BLINKS= {}".format(len(retrieved_blinks)))
                            return retrieved_blinks,int(TOTAL_BLINKS),Counter4blinks,BLINK_READY,skip
                        else:
                            skip=True
                            print('rejected due to imbalance')
                    else:
                        skip = True
                        print('rejected due to noise,magnitude is {}'.format(Last_Blink.amplitude))
                        print(Last_Blink.start<Last_Blink.peak)
            if Counter4blinks >1:
                Current_Blink.end = reference_frame - 7  #reference-7 points to the last frame that eyes were closed
                Current_Blink.endEAR=Current_Blink.EAR_of_FOI
                Current_Blink.amplitude = (Current_Blink.startEAR + Current_Blink.endEAR - 2 * Current_Blink.peakEAR) / 2
                Current_Blink.duration = Current_Blink.end - Current_Blink.start + 1
                if Last_Blink.duration>15:
                    FRAME_MARGIN_BTW_2BLINKS=8
                else:
                    FRAME_MARGIN_BTW_2BLINKS=1
                if (Current_Blink.start-Last_Blink.end )<=FRAME_MARGIN_BTW_2BLINKS+1:  #Merging two close blinks
                    print('Merging...')
                    frames_in_between=Current_Blink.start - Last_Blink.end-1
                    print(Current_Blink.start ,Last_Blink.end, frames_in_between)
                    valuesBTW=Linear_Interpolate(Last_Blink.endEAR,Current_Blink.startEAR,frames_in_between)
                    Last_Blink.values=Last_Blink.values+valuesBTW+Current_Blink.values
                    Last_Blink.end = Current_Blink.end            # update the end
                    Last_Blink.endEAR = Current_Blink.endEAR
                    if Last_Blink.peakEAR>Current_Blink.peakEAR:  #update the peak
                        Last_Blink.peakEAR=Current_Blink.peakEAR
                        Last_Blink.peak = Current_Blink.peak
                    Last_Blink.amplitude = (Last_Blink.startEAR + Last_Blink.endEAR - 2 * Last_Blink.peakEAR) / 2
                    Last_Blink.duration = Last_Blink.end - Last_Blink.start + 1
                else:                                             #Should not Merge (a Separate blink)
                    Last_Blink.values=Current_Blink.values        #update the EAR list
                    Last_Blink.end = Current_Blink.end            # update the end
                    Last_Blink.endEAR = Current_Blink.endEAR
                    Last_Blink.start = Current_Blink.start        #update the start
                    Last_Blink.startEAR = Current_Blink.startEAR
                    Last_Blink.peakEAR = Current_Blink.peakEAR    #update the peak
                    Last_Blink.peak = Current_Blink.peak
                    Last_Blink.amplitude = Current_Blink.amplitude
                    Last_Blink.duration = Current_Blink.duration
            Counter4blinks = 0
        retrieved_blinks=0
        return retrieved_blinks,int(TOTAL_BLINKS),Counter4blinks,BLINK_READY,skip
    COUNTER = 0
    MCOUNTER=0
    TOTAL = 0
    MTOTAL=0
    TOTAL_BLINKS=0
    Counter4blinks=0
    skip=False # to make sure a blink is not counted twice in the Blink_Tracker function
    Last_Blink=Blink()
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    loaded_svm = pickle.load(open('Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    print("[INFO] starting video stream thread...")
    lk_params=dict( winSize  = (13,13),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
    global u_features
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
        rects = detector(gray, 0)

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
    stream.release()
    cv2.destroyAllWindows()




output_file = 'alert.txt'  # The text file to write to (for blinks)#
path = '../Fold3_part2/31/10.mp4' # the path to the input video

blink_detector(output_file,path)
f = open("10input.txt","w")
f.write(str(final))
f.close()
