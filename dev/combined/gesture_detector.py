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
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


class GestureDetector:
    """
    Detect emotions in a given image of a human face.
    얼굴 표정을 인식하는 모델을 가져오고 표정을 인식한다.
    """
    def __init__(self, model_path=None):
        # MediaPipe hands model
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.actions = ['ON', 'OFF', 'Blur1', 'Blur2', 'Blur3',
                        'emoticon_ON', 'emoticon_OFF']
        self.seq_length = 30
        self.model = load_model(model_path)  # 손동작 인식용 모델을 로딩

    def detect_gesture(self, cap, seq, action_seq, roi):
        """
        Function to detect emotions in a human face image.
        """
        img = roi
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        g_flag = "UNKNOWN"
        b_value = -1

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                seq.append(self.joint_compute(res))

                self.mp_drawing.draw_landmarks(img, res,
                                               self.mp_hands.HAND_CONNECTIONS)

                if len(seq) < self.seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-self.seq_length:],
                                                     dtype=np.float32), axis=0)

                y_pred = self.model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = self.actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue
                
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    # Perform action based on the recognized gesture
                    g_flag, b_value = self.switch_case(action)

        return g_flag, b_value

    def switch_case(self, action):
        """
        Hand gesture switch case.
        """
        b_value = -1

        if action == 'ON':
            action = "DEFAULT"
        elif action == 'OFF':
            action = "NONE"
        elif action == 'Blur1':
            action = "DEFAULT"
            b_value = 0
        elif action == 'Blur2':
            action = "DEFAULT"
            b_value = 20
        elif action == 'Blur3':
            action = "DEFAULT"
            b_value = 50
        elif action == 'emoticon_ON':
            action = "CHANGE"
        elif action == 'emoticon_OFF':
            action = "DEFAULT"

        return action, b_value

    def joint_compute(self, res):
        """
        Compute angles between joints
        """
        joint = np.zeros((21, 4))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13,
                    14, 15, 0, 17, 18, 19], :3]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20], :3]  # Child joint
        v = v2 - v1  # [20, 3]
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize v

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                       12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11,
                                       13, 14, 15, 17, 18, 19], :]))
        angle = np.degrees(angle)  # Convert radian to degree

        return np.concatenate([joint.flatten(), angle])
