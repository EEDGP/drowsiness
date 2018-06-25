from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import ctypes
import math
import os
import time


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:


				EV_SND = 0x12  # linux/input-event-codes.h
				SND_TONE = 0x2  # ditto
				time_t = suseconds_t = ctypes.c_long

				class Timeval(ctypes.Structure):
				    _fields_ = [('tv_sec', time_t),       # seconds
				                ('tv_usec', suseconds_t)] # microseconds

				class InputEvent(ctypes.Structure):
				    _fields_ = [('time', Timeval),
				                ('type', ctypes.c_uint16),
				                ('code', ctypes.c_uint16),
				                ('value', ctypes.c_int32)]


				frequency = 440  # Hz, A440 in ISO 16
				device = "/dev/input/by-path/platform-pcspkr-event-spkr"
				pcspkr_fd = os.open(device, os.O_WRONLY)  # root! + modprobe pcspkr
				fsec, sec = math.modf(time.time())  # current time
				ev = InputEvent(time=Timeval(tv_sec=int(sec), tv_usec=int(fsec * 1000000)),
				                type=EV_SND,
				                code=SND_TONE,
				                value=frequency)
				os.write(pcspkr_fd, ev)  # start beep
				try:
				    time.sleep(0.2)  # 200 milliseconds
				finally:
				    ev.value = 0  # stop
				    os.write(pcspkr_fd, ev)

		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
