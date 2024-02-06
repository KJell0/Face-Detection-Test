import cv2
#trained xml file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#input image
image = cv2.imread('test.jpg')
#convert image into greyscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#rectangle for faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


cv2.imshow('img', image)
cv2.waitKey()