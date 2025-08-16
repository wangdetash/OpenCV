import cv2
import datetime

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)  # can either provide path to file name or device index which can be 0 or -1
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3, 640)  # if  invalid camera resolution is fed, the values wont change but do not drop  any error.
cap.set(4, 480)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "output.avi", fourcc, 20.0, (640, 480)
)  # fourcc code is a four byte code which is used to specify the video codac. 20fps (640*480)  capture size

events = [i for i in dir(cv2) if "EVENT" in i]
print(events)

while cap.isOpened():  # checkin if the video can be accessed
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a blue square around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the frame as a JPEG image
    cv2.imwrite("captured_image.jpg", frame)

    out.write(frame)  # write the file

    datetime_text = str(datetime.datetime.now())
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_size_text = "Width:" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + " " + "Height:" + str(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    frame = cv2.putText(frame, frame_size_text, (10, 50), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, datetime_text, (10, 75), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()