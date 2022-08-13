import cv2
bodyclassifier=cv2.CascadeClassifier('PRO-C106-ProjectSolution-main/haarcascade_fullbody.xml')
video=cv2.VideoCapture('PRO-C106-ProjectSolution-main/walking.avi')
while True:
    ret,img=video.read()
    grays=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bodies=bodyclassifier.detectMultiScale(grays,1.2,3)
    for(x,y,w,h) in bodies:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('walking',img)
    if (cv2.waitKey(1)==32):
        break    
video.release() 
cv2.destroyAllWindows()   