import cv2
import datetime
import numpy as np
import time
import sys
import argparse
from openvino.runtime import Core

def parse_arguments():
    parser = argparse.ArgumentParser(description='Face detection with OpenVINO')
    parser.add_argument('--device', type=str, choices=['CPU', 'NPU'], default='NPU',
                       help='Device to run inference on: CPU or NPU (default: NPU)')
    return parser.parse_args()

def setup_model(device_name):
    core = Core()
    
    # Select model precision based on device
    if device_name == "NPU":
        model_path = "./intel/face-detection-adas-0001/FP16-INT8/face-detection-adas-0001.xml"
        device_config = {
            "NPU_DMA_ENGINES": "2",  # Use dual DMA engines for parallel data transfer
        }
        print(f"Loading FP16-INT8 quantized model for {device_name}")
    else:  # CPU
        model_path = "./intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
        device_config = {}
        print(f"Loading FP16 model for {device_name}")
    
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device_name=device_name, config=device_config)
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Create inference request
    infer_request = compiled_model.create_infer_request()
    
    print(f"Model compiled for {device_name} device")
    return compiled_model, input_layer, output_layer, infer_request

# Parse command line arguments
args = parse_arguments()
device = args.device

# Setup model based on selected device
compiled_model, input_layer, output_layer, infer_request = setup_model(device)
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

    # Preprocess frame for the OpenVINO model with NPU-optimized preprocessing
    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 1] range for better NPU performance
    normalized = resized.astype(np.float32) / 255.0
    input_image = normalized.transpose((2, 0, 1))[np.newaxis, :]

    # Run inference using async request for better performance
    start_time = time.perf_counter()
    infer_request.infer([input_image])
    detections = infer_request.get_output_tensor(0).data
    inference_ms = (time.perf_counter() - start_time) * 1000
    print(f"{device} Inference time: {inference_ms:.2f} ms")

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

    # Add inference time and processor info at top right in red
    font = cv2.FONT_HERSHEY_TRIPLEX  # Closest to Times New Roman in OpenCV
    inference_text = f"Inference time: {inference_ms:.1f}ms {device}"
    text_size = cv2.getTextSize(inference_text, font, 0.7, 1)[0]
    text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from right edge
    text_y = 30  # 30 pixels from top
    frame = cv2.putText(frame, inference_text, (text_x, text_y), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    
    out.write(frame)  # write the file
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
