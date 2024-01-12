"""
Class to detect emotions in the image of a human face.
Uses an emotion detection model from Intel's Open Model Zoo.
Ref: https://docs.openvino.ai/2023.2/\
    omz_models_model_emotions_recognition_retail_0003.html

사람 얼굴에서 감정을 인식하기 위한 클래스.
인텔 오픈비노(Openvino)의 오픈모델주(Open Model Zoo)에 있는
감정인식 머신러닝/인공지능 모델을 활용한다.
위 링크에서 모델 정보를 추가로 찾을 수 있다.
"""
import collections
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


class PersonDetector:
    """
    Detect people in a given image.
    영상을 가져와서 사람을 인식하는 모델로 사람을 인식한다.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.device = "CPU"
        self.confidence = 0.5
        self.core = ov.Core()

        # Read the network and corresponding weights from a file.
        self.model = self.core.read_model(model=self.model_path)
        # Compile the model for CPU (you can choose manually CPU, GPU etc.)
        # or let the engine choose the best available device (AUTO).
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device)

        # Get the input and output nodes.
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Get the input size.
        self.height, self.width = list(self.input_layer.shape)[1:3]

        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        self.classes = [
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
            "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush", "hair brush"
        ]


    def process_results(self, frame, results, thresh=0.8):
        """
        Function to detect people in an image.
        """
        # The size of the original frame.
        h, w = frame.shape[:2]
        # The 'results' variable is a [1, 1, 100, 7] tensor.
        results = results.squeeze()
        boxes = []
        labels = []
        scores = []
        for _, label, score, xmin, ymin, xmax, ymax in results:
            if label == 1:
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
                boxes.append(
                    tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
                )
                labels.append(int(label))
                scores.append(float(score))

        # Apply non-maximum suppression to get rid of many overlapping entities.
        # See https://paperswithcode.com/method/non-maximum-suppression
        # This algorithm returns indices of objects to keep.
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
        )

        # If there are no boxes.
        if len(indices) == 0:
            return []

        # Filter detected objects.
        return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

    def detect_people(self, frame):
        # Resize the image and change dims to fit neural network input.
        input_img = cv2.resize(
            src=frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA
        )
        # Create a batch of images (size = 1).
        input_img = input_img[np.newaxis, ...]

        # Get the results.
        results = self.compiled_model([input_img])[self.output_layer]
        # Get poses from network results.
        boxes = self.process_results(frame=frame, results=results)

        # cv2.imshow(winname="ret", mat=frame)
        # cv2.waitKey(1)

        return boxes
