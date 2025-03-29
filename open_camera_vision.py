import cv2

cap = cv2.VideoCapture(0) #can either provide path to file name or device index which can be 0 or -1

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()