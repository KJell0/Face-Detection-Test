import cv2
#Trained xml file from Haarcascades (face and eyes)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#Takes video from your webcam
cap = cv2.VideoCapture(0)

#Main loop to keep detecting
while True:
    #Read frames
    _, img = cap.read()

    #Conversion to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    #Drawing rectangles for the face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    #Detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray)

    #Drawing rectangles for the eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    #Display
    cv2.imshow('img', img)

    #Exit with ESC
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()