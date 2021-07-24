from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from threading import Thread
import imutils
import numpy as np
from pyzbar.pyzbar import decode


camera = PiCamera()
camera.resolution = (800, 608)
camera.framerate = 32
rawCapture1 = PiRGBArray(camera, size=(800, 608))
rawCapture2 = PiRGBArray(camera, size=(800, 608))
BackSub = cv2.createBackgroundSubtractorMOG2(500, 21, True)
FirstFrame=None
DetCount = 0
FrameCount = 0

def Motion_Camera(rawCaptureGrey, QRDetect):
    for f in camera.capture_continuous(rawCaptureGrey, format="bgr", use_video_port=True):
        frame = f.array
        rawCapture1.truncate(0)
        if frame is not None:
            if QRDetect:
                return frame
            else:
                frame = BackSub.apply(frame)
                return frame
            
def QR_Camera(rawCaptureGrey, QRDetect):
    for f in camera.capture_continuous(rawCaptureGrey, format="bgr", use_video_port=True):
        frame = f.array
        rawCapture2.truncate(0)
        if frame is not None:
            return frame

def Motion_Detection(Grey_Frame):
    Det_Index = False
    thresh = cv2.threshold(Grey_Frame, 230, 255, cv2.THRESH_BINARY)[1]
#    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (5,5))
    '''cv2.imshow("frame", thresh)
    if cv2.waitKey(1) == ord('q'):
        exit()
    return thresh'''
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) > 10000:
            Det_Index=True
#            return thresh
            return Det_Index
        
def get_qr_data(input_frame):
    try:
        return decode(input_frame)
    except:
        return []

def draw_polygon(f_in, qro):
    if len(qro) == 0:
        return f_in
    else:
        for obj in qro:
            text = obj.data.decode('utf-8')
            pts = np.array([obj.polygon],np.int32)
            pts = pts.reshape((4,1,2))
            cv2.polylines(f_in, [pts], True, (255, 100, 5), 2)
            cv2.putText(f_in, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 100, 5), 2)
        return f_in

while True:
    f = Motion_Camera(rawCapture1, False)
    DetVar = False
#    (DetVar, frame) = Motion_Detection(f)
    DetVar = Motion_Detection(f)
    if DetVar:
        DetCount += 1
        print(DetCount)
        print(DetVar)
        if DetCount == 1:
            continue
        else:
            while DetVar:
                QR_f = QR_Camera(rawCapture2, True)
                qr_obj = get_qr_data(QR_f)
                frame = draw_polygon(f, qr_obj)
                cv2.imshow("frame", frame)
                if cv2.waitKey(3000):
                    DetVar = False
                    break

cv2.destroyAllWindows()
    

