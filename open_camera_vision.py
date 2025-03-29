import cv2

cap = cv2.VideoCapture(0) #can either provide path to file name or device index which can be 0 or -1

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()

