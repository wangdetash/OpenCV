import cv2

cap = cv2.VideoCapture(0) #can either provide path to file name or device index which can be 0 or -1
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out =  cv2.VideoWriter('output.avi',fourcc,20.0,(640,480)) #fourcc code is a four byte code which is used to specify the video codac. 20fps (640*480)  capture size
while(cap.isOpened()): #checkin if the video can be accessed
    ret, frame = cap.read()

    out.write(frame)#write the file 

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()