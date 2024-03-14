import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

hood_image = cv2.imread("deco/deco_crocodile.png", cv2.IMREAD_UNCHANGED)
hood_image = cv2.resize(hood_image, (640, 480))
alpha_mask = hood_image[:, :, 3] / 255.0
hood_image = hood_image[:, :, :3]

while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        hood_width = face_landmarks.part(16).x - face_landmarks.part(0).x
        hood_height = int(hood_width * (hood_image.shape[0] / hood_image.shape[1]))
        hood_resized = cv2.resize(hood_image, (hood_width, hood_height))
        middle_x = (face_landmarks.part(16).x + face_landmarks.part(0).x) // 2
        middle_y = (face_landmarks.part(16).y + face_landmarks.part(0).y) // 2
        print("hood_resized shape:", hood_resized.shape)
        hood_x = middle_x - (hood_width // 2)
        hood_y = middle_y - (hood_height // 2)
        alpha_mask_resized = cv2.resize(alpha_mask, (hood_width, hood_height))
        alpha_mask_resized = np.atleast_3d(alpha_mask_resized)

        bg = frame[hood_y:hood_y + hood_height, hood_x:hood_x + hood_width]
        bg = bg.astype(np.float32)
        np.multiply(bg, 1 - alpha_mask_resized, out=bg, casting="unsafe")
        np.add(bg, hood_resized * alpha_mask_resized, out=bg)

        frame[hood_y:hood_y + hood_height, hood_x:hood_x + hood_width] = bg

        print("hood_resized shape:", hood_resized.shape)
        print("frame region shape:", frame[hood_y:hood_y + hood_height, hood_x:hood_x + hood_width].shape)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
