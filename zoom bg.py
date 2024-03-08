import cv2
import dlib
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 40)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
segmentor = SelfiSegmentation()
imgBG = cv2.imread("froest.jpg")
imgBG = cv2.resize(imgBG, (640, 480))

while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgOut = segmentor.removeBG(frame, imgBG, cutThreshold=0.9)


    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    #imgStacked = cvzone.stackImages([frame,imgOut],2,1)
    cv2.imshow("Face Landmarks", frame)

    cv2.imshow("image out", imgOut)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
