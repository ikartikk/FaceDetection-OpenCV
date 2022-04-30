import cv2 as cv
from cv2 import CascadeClassifier

cap = cv.VideoCapture(0)
Cascade_Classifier=cv.CascadeClassifier('computer_vision\haarcascades\haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()

    frame= cv.cvtColor(frame,0)
    detection = Cascade_Classifier.detectMultiScale(frame)

    if(len(detection)>0):
        (x,y,w,h)=detection[0]
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


