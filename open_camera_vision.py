import cv2
import datetime

cap = cv2.VideoCapture(0) #can either provide path to file name or device index which can be 0 or -1
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3,640) #if  invalid camera resolution is fed, the values wont change but do not drop  any error.
cap.set(4,480)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out =  cv2.VideoWriter('output.avi',fourcc,20.0,(640,480)) #fourcc code is a four byte code which is used to specify the video codac. 20fps (640*480)  capture size
while(cap.isOpened()): #checkin if the video can be accessed
    ret, frame = cap.read()

    out.write(frame)#write the file 

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    datetime_text= str(datetime.datetime.now())
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_size_text = 'Width:' + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + ' ' + 'Height:' + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gray = cv2.putText(gray,frame_size_text,(10,50),font,1,(0,255,255),1,cv2.LINE_AA)
    gray = cv2.putText(gray,datetime_text,(10,75),font,1,(0,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()