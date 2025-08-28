import cv2
import datetime
import numpy as np
import time
from openvino.runtime import Core

# Load the OpenVINO face detection model and compile for Intel NPU
core = Core()
# Replace the path below with the location of your OpenVINO IR (.xml) model
model = core.read_model("face-detection-adas-0001.xml")
compiled_model = core.compile_model(model, device_name="NPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
# Get the expected input resolution for preprocessing
_, _, h, w = input_layer.shape

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
        print("Failed to grab frame")
        break

    # Preprocess frame for the OpenVINO model
    resized = cv2.resize(frame, (w, h))
    input_image = resized.transpose((2, 0, 1))[np.newaxis, :]

    # Run inference on the Intel NPU and report latency
    start_time = time.perf_counter()
    detections = compiled_model([input_image])[output_layer]
    inference_ms = (time.perf_counter() - start_time) * 1000
    print(f"Inference time: {inference_ms:.2f} ms")

    # Draw a blue square around each detected face
    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

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
