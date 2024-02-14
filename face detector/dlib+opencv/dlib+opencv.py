import cv2
import dlib

#Create a VideoCapture object. 
#This VideoCapture object will be connected to a network camera. 
#We can use its parameters to specify which camera to use.
#(0 represents the first one, 1 represents the second one).
cap = cv2.VideoCapture(0)
#Returns(build) the default face detector
hog_face_detector = dlib.get_frontal_face_detector()
#takes in an image region containing some object 
#and outputs a set of point locations that define the pose of the object.
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:

    #read a frame from a camera or video file. 
    #It will return two values,
    #first is a Boolean value, indicating whether the image frame is successfully read, 
    #second is the read image frame.
    #Use the variable name _ to ignore the first return value, store second value in name frame
    _, frame = cap.read()

    #frame: The original image to be converted.
    #cv2.COLOR_BGR2GRAY: from BGR to grayscale.
    #Ad: Simplify image processing calculations, 
    #reduce resource usage or requirements in specific application areas.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Pass the grayscale image gray to hog_face_detector for face detection, and 
    #store the detection results in a variable named faces. 
    faces = hog_face_detector(gray)
    for face in faces:
        #Detect the feature points of each face in the grayscale image
        #store the results in the face_landmarks variable for subsequent use.
        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            #Store the x(y)-coordinate of the feature point in the x(y) variable
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            #Use the cv2.circle() function to draw a circle on the frame with the center coordinates (x, y) and a radius of 1. 
            #The line color is (0, 255, 255) (blue, green, red, or yellow)
            #the line width is 1.
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
