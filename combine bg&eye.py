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
imgBG = cv2.imread("bg/bg_deer.png")
imgBG = cv2.resize(imgBG, (640, 480))
eye_image = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
eye_image = eye_image[:, :, :3]

while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = segmentor.removeBG(frame, imgBG)


    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # Overlay eye image on face landmarks
        left_eye_width = face_landmarks.part(40).x - face_landmarks.part(37).x
        left_eye_height = int(left_eye_width * (eye_image.shape[0] / eye_image.shape[1]))
        left_eye_image_resized = cv2.resize(eye_image, (left_eye_width, left_eye_height))
        left_eye_x = face_landmarks.part(37).x
        left_eye_y = face_landmarks.part(37).y
        frame[left_eye_y:left_eye_y + left_eye_height, left_eye_x:left_eye_x + left_eye_width] = left_eye_image_resized
        right_eye_image_resized = cv2.resize(eye_image, (left_eye_width, left_eye_height))
        right_eye_x = face_landmarks.part(43).x
        right_eye_y = face_landmarks.part(43).y
        frame[right_eye_y:right_eye_y + left_eye_height, right_eye_x:right_eye_x + left_eye_width] = right_eye_image_resized


    #imgStacked = cvzone.stackImages([frame,imgOut],2,1)
    cv2.imshow("Face Landmarks", frame)

    #cv2.imshow("image out", imgOut)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
