import cv2
import dlib
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 40)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
segmentor = SelfiSegmentation()
imgBG = cv2.imread("bg/bg_deer.png")
imgBG = cv2.resize(imgBG, (640, 480))

def place_eyes():
    #input
    eye_image = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
    eye_image = eye_image[:, :, :3]
    #
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

def place_deco():
    #input
    scale_factor = 1.6
    hood_image = cv2.imread("deco/deco_crocodile.png", cv2.IMREAD_UNCHANGED)
    hood_image = cv2.resize(hood_image, (640, 480))
    alpha_mask = hood_image[:, :, 3] / 255.0
    hood_image = hood_image[:, :, :3]
    #
    hood_width = int((face_landmarks.part(16).x - face_landmarks.part(0).x) * scale_factor)
    hood_height = int(hood_width * (hood_image.shape[0] / hood_image.shape[1]))
    hood_resized = cv2.resize(hood_image, (hood_width, hood_height))
    middle_x = (face_landmarks.part(15).x + face_landmarks.part(1).x) // 2
    middle_y = (face_landmarks.part(15).y + face_landmarks.part(1).y) // 2
    hood_x = middle_x - (hood_width // 2)
    hood_y = middle_y - (hood_height // 2)
    alpha_mask_resized = cv2.resize(alpha_mask, (hood_width, hood_height))
    alpha_mask_resized = np.atleast_3d(alpha_mask_resized)

    bg = frame[hood_y:hood_y + hood_height, hood_x:hood_x + hood_width]
    bg = bg.astype(np.float32)
    np.multiply(bg, 1 - alpha_mask_resized, out=bg, casting="unsafe")
    np.add(bg, hood_resized * alpha_mask_resized, out=bg)

    frame[hood_y:hood_y + hood_height, hood_x:hood_x + hood_width] = bg

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

        place_eyes()
        place_deco()

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
