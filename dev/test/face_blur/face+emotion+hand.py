#!/usr/bin/env python3
"""
 Copyright (c) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Face detection imports
import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.runtime import Core, get_version

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

# 231230 투명한 배경의 이미지를 이용하기 위해서 사용하는 라이브러리 명령창에서 꼭 다운받아야 한다. "pip install pillow"
from PIL import Image

# Hand gesture detection imports
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']

actions = ['ON', 'OFF', 'Blur1', 'Blur2', 'Blur3', 'emoticon_ON', 'emoticon_OFF']
seq_length = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

seq = []
action_seq = []


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=True,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='', help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=True,
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('-m_ed', type=Path, required=True,
                        help='Required. Path to an .xml file with Emotion Detection model.')
    models.add_argument('-m_gd', type=Path, required=True,
                        help='Required. Path to a TensorFlow model .h5 file with Gesture Detection model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
                            
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    return parser


### emotion switch case
def switch_case(value):
    if value == 0: #nuetral
        return './emotion/neutral.png'
    elif value == 1: #happy
        return  './emotion/happy.png'
    elif value == 2: #sad
        return './emotion/sad.png'
    elif value == 3: #surprised
        return './emotion/surprised.png'
    elif value == 4: #angry
        return  './emotion/angry.png'
    else: #default: nuetral
        return  './emotion/neutral.png'


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)
    
        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]


# 231229
class EmotionDetector:
    # 231229  얼굴 표정을 인식하는 모델을 가져오고
    def __init__(self, model_path=None):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
    # 231229  표정 인식하는 함수
    def detect_emotion(self, face_block):
        resized_image = cv2.resize(face_block, (64, 64))
        input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        result = self.compiled_model(input_data)[self.output_layer]
        flat_result = result.flatten()
        max_index = np.argmax(flat_result)
        emotion = switch_case(max_index)
        return emotion # 해당 표정에 맞는 이모지를 주소로 가져옴 
    

#  231230 
class FaceDrawer:
    @staticmethod
    # 인식한 얼굴 위에 이모티콘 이미지를 붙인다. 
    def draw_face(frame, xmin, ymin, xmax, ymax, text, landmarks, output_transform, emotion_image):
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale((xmax - xmin) * point[0])
            y = ymin + output_transform.scale((ymax - ymin) * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)

        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        
        # 감정 이미지를 얼굴 영역의 높이에 맞추어 크기 조절
        emotion_image_resized = cv2.resize(emotion_image, (int(xmax - xmin), int(ymax - ymin)))

        # 이미지 합성
        frame = FaceDrawer.overlay_transparent(frame, emotion_image_resized, x=xmin, y=ymin)
        

        # 이모티콘을 붙인 얼굴 프레임 
        return frame 

    #투명한 배경을 그대로 이용하기 위해서 알파값을 살려 투명한 부분에 오버레이되게 한다.  
    @staticmethod 
    def overlay_transparent(background, overlay, x, y):
        """
        Overlay a transparent image onto a background.

        Parameters:
        - background: Background image.
        - overlay: Overlay image with transparency (RGBA format).
        - x, y: Coordinates to place the overlay image.

        Returns:
        - Resulting image after overlay.
        """
        background = background.copy()

        # Extract the alpha channel from the overlay image
        overlay_alpha = overlay[:, :, 3] / 255.0

        # Calculate the inverse of the alpha channel
        background_alpha = 1.0 - overlay_alpha

        # Calculate the weighted sum of the background and overlay images
        for c in range(0, 3):
            background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = \
                (background_alpha * background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] +
                 overlay_alpha * overlay[:, :, c])

        return background

   
# 231230 
def draw_detections(frame, frame_processor, detections, output_transform, blur_mode, args):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)

    emotion_detector = EmotionDetector(args.m_ed)

    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])

        face_block = frame[ymin:ymax, xmin:xmax]

        if(blur_mode == "DEFAULT"):
            blur = cv2.GaussianBlur(face_block, (51, 51), 0)
            frame[ymin:ymax, xmin:xmax] = blur
            print("this is defualt")
        elif(blur_mode == "NONE"):
            print("this is none")
            pass
        elif (blur_mode == "CHANGE"):
            print("this is change")
            # 감정 이미지를 얼굴 영역의 높이에 맞추어 크기 조절
            emotion = emotion_detector.detect_emotion(face_block)
            bgr_image= cv2.imread(emotion, cv2.IMREAD_UNCHANGED)
            emotion_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
            frame = FaceDrawer.draw_face(frame, xmin, ymin, xmax, ymax, text, landmarks, output_transform, emotion_image)

    return frame


def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    frame_processor = FrameProcessor(args)

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    
    model = load_model(args.m_gd)  # 손동작 인식용 모델

    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()
    
    # 231230 인터럽트로 필터를 끄고 킬 용도의 변수
    blur_mode = "DEFAULT"

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)
        presenter.drawGraphs(frame)

        # 240106 손동작 인식 모델
        img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action
                    print(this_action)

                    # Perform action based on the recognized gesture
                    if this_action == 'ON':
                        blur_mode = "DEFAULT"
                    elif this_action == 'OFF':
                        blur_mode = "NONE"
                    elif this_action == 'Blur1':
                        blur_mode
                    elif this_action == 'Blur2':
                        blur_mode
                    elif this_action == 'Blur3':
                        blur_mode
                    elif this_action == 'emoticon_ON':
                        blur_mode = "CHANGE"
                    elif this_action == 'emoticon_OFF':
                        blur_mode = "DEFAULT"

                cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        frame = draw_detections(frame, frame_processor, detections, output_transform, blur_mode, args)
        frame_num += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # if key in {ord('x'), ord('X')}:  # 'x' 키를 누르면 blur_mode를 'NONE'으로 변경
            #     blur_mode = "NONE"
            #     print (f"key: {key}")
            #     print(f"blur mod: {blur_mode}")
            # elif key in {ord('c'), ord('C')}:  # 'c' 키를 누르면 blur_mode를 'CHANGE'로 변경
            #     blur_mode = "CHANGE"
            #     print (f"key: {key}")
            #     print(f"blur mod: {blur_mode}")
            # elif key in {ord('o'), ord('O')}:  # 'O' 키를 누르면 blur_mode를 'DEFAULT'로 변경
            #     blur_mode = "DEFAULT"
            #     print (f"key: {key}")
            #     print(f"blur mod: {blur_mode}")
            if key in {ord('q'), ord('Q'), 27}:  # 'q' 키 또는 ESC를 누르면 종료
                break
            presenter.handleKey(key)

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
