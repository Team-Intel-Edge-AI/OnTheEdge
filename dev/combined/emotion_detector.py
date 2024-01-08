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

from openvino.runtime import Core
import cv2
import numpy as np


class EmotionDetector:
    """
    Detect emotions in a given image of a human face.
    얼굴 표정을 인식하는 모델을 가져오고 표정을 인식한다.
    """
    def __init__(self, model_path=None):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model=self.model,
                                                      device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def detect_emotion(self, face_block):
        """
        Function to detect emotions in a human face image.
        """
        resized_image = cv2.resize(face_block, (64, 64))
        input_data = np.expand_dims(
            np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        result = self.compiled_model(input_data)[self.output_layer]
        flat_result = result.flatten()
        max_index = np.argmax(flat_result)
        emotion = self.switch_case(max_index)
        return emotion  # 해당 표정에 맞는 이모지를 주소로 가져옴

    def switch_case(self, value):
        """
        Emotion switch case.
        """
        if value == 0:  # nuetral
            return './emotion/neutral.png'
        if value == 1:  # happy
            return './emotion/happy.png'
        if value == 2:  # sad
            return './emotion/sad.png'
        if value == 3:  # surprised
            return './emotion/surprised.png'
        if value == 4:  # angry
            return './emotion/angry.png'
        return './emotion/neutral.png'  # default: nuetral
