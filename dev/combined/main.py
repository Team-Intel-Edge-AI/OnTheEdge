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
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[0] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[0]
                    / 'common/python/openvino/model_zoo'))

import monitors
from helpers import resolution
from images_capture import open_images_capture
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

from face_identifier import FaceIdentifier
from frame_processor import FrameProcessor
from face_drawer import FaceDrawer
from emotion_detector import EmotionDetector
from gesture_detector import GestureDetector
from person_detector import PersonDetector

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def build_argparser():
    """
    Parse command line arguments and define 'help' output.

    터미널 입력을 위한 매개변수를 정의하고, 실제로 실행했을 때 입력 받는 값을 정의한다.
    'help' 도움말 출력값 또한 명시한다.
    """
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=True,
                         help='Required. An input to process. The input must '
                              'be a single image, a folder of images, '
                              'video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save. \
                         Frames of odd width or height can be truncated. \
                         See https://github.com/opencv/opencv/pull/24086')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window \
                            resolution in (width x height) format. \
                            Example: 1280x720. Input frame size used \
                            by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to \
                            this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN',
                         choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. \
                            Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')
    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='', help='Optional.\
                         Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find \
                            faces on the face images, otherwise use \
                            full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and \
                            to dump on disk. Available only if --no_show \
                            option is off.')
    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=True,
                        help='Required. \
                            Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=True,
                        help='Required. Path to an .xml file with \
                            Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=True,
                        help='Required. Path to an .xml file with \
                            Face Reidentification model.')
    models.add_argument('-m_ed', type=Path, required=True,
                        help='Required. Path to an .xml file with \
                            Emotion Detection model.')
    models.add_argument('-m_gd', type=Path, required=True,
                        help='Required. Path to a TensorFlow model .h5 file \
                            with Gesture Detection model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection \
                            model for reshaping. Example: 500 700.')
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model.\
                            Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks \
                        Detection model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face \
                        Reidentification model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for \
                        face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between \
                        two vectors for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for boxes \
                        passed to face recognition.')
    person = parser.add_argument_group('Person detection')
    person.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5m.onnx',
                        help='model path or triton URL')
    person.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    person.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml',
                        help='(optional) dataset.yaml path')
    person.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int,
                        default=[640], help='inference size h,w')
    person.add_argument('--conf-thres', type=float, default=0.7,
                        help='confidence threshold')
    person.add_argument('--hide-conf', default=False, action='store_true',
                        help='hide confidences')

    return parser


def draw_detections(frame, frame_processor, detections,
                    output_transform, blur_mode, emotion_detector):
    """
    Draw detections.
    """
    print("INSIDE DRAW_DETECTIONS")

    size = frame.shape[:2]
    frame = output_transform.resize(frame)

    print("BEFORE LOOP IN DRAW_DETECTIONS")
    print(detections)
    for roi, landmarks, identity in zip(*detections):
        print("LOOPING INSIDE DRAW_DETECTIONS")
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin,
                                                         xmax, ymax])

        face_block = frame[ymin:ymax, xmin:xmax]

        print(identity.id)
        print(blur_mode)

        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        else:  # if identity.id == FaceIdentifier.UNKNOWN_ID
            if blur_mode == "DEFAULT":
                frame[ymin:ymax, xmin:xmax] = cv2.GaussianBlur(
                    face_block, (51, 51), FrameProcessor.blur_value)
            elif blur_mode == "CHANGE":
                # 감정 이미지를 얼굴 영역의 높이에 맞추어 크기 조절
                emotion = emotion_detector.detect_emotion(face_block)
                bgr_image = cv2.imread(emotion, cv2.IMREAD_UNCHANGED)
                emotion_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
                frame = FaceDrawer.draw_face(frame, xmin, ymin, xmax, ymax,
                                             text, landmarks, output_transform,
                                             emotion_image)
            else:  # if blur_mode == "NONE"
                pass

        # Uncomment to display text on image, 영상에 텍스트를 출력하기 위해서는 아래 주석을 푸시오.
        # textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        # cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin -
        #               textsize[1]), (0, 255, 0), cv2.FILLED)
        # cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (0, 0, 0), 1)

    return frame


def center_crop(frame, crop_size):
    """
    Center crop.
    """
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2: (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2: (fw + crop_size[0]) // 2,
                 :]


def main():
    """
    Main function, executed by running the run.sh file.
    Do not separately execute this function or this file,
    because external ML/AI models have to be provided for execution.
    The paths to these models are specified in the run.sh file.

    Execute in this directory in the command-line with "./run.sh"

    main 함수나 main 파일만 따로 실행하지 마세요.
    이 코드를 실행하기 위해서는 외부에서 머신러닝/인공지능 모델을 전달해야하는데,
    'run.sh' 파일에서 해당 모델들을 전달하기에 'run.sh'파일을 실행해야 합니다.

    이 폴더를 터미널에서 열어서 "./run.sh" 명령을 실행하세요.
    """
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    frame_processor = FrameProcessor(args)
    gesture_detector = GestureDetector(args.m_gd)
    emotion_detector = EmotionDetector(args.m_ed)
    person_detector = PersonDetector()
    
    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None

    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    seq = []
    action_seq = []

    # *** 손동작 인식으로 값이 변하는 제어 변수 ***
    m_flag = "DEFAULT"  # 기본값으로 얼굴 blurring 활성화


    while True:
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = \
                OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = \
                monitors.Presenter(args.utilization_monitors, 55,
                                   (round(output_resolution[0] / 4),
                                    round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(
                args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                    cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        predictions, img_roi = person_detector.run(frame, args.weights, args.source)

        for tup in predictions:
            points = tup[2]
            print("PRINTING POINTS, ROI, FRAME")
            # roi = img_roi[points[1]:points[3]+400, points[0]:points[2]+400]

            cropped = frame.copy()
            for x,y in np.ndindex(frame.shape[:2]):
                if y < points[1] or y > points[3]+400 or x < points[0] or x > points[2]+400:
                    cropped[x][y] = [0, 0, 0]

            # cv2.imshow('Test', roi)
            # key = cv2.waitKey(1)
            # if key in {ord('q'), ord('Q'), 27}:  # 'q' 키 또는 ESC를 누르면 종료
            #     break

            detections = frame_processor.process(cropped)
            presenter.drawGraphs(cropped)

            print("BEFORE GESTURE DETECTION")
            g_flag, b_value = gesture_detector.detect_gesture(cap, seq, action_seq, cropped)
            print(g_flag)
            print(b_value)
            print("AFTER GESTURE DETECTION")
            if g_flag != "UNKNOWN":  # g_flag == "" if no change
                m_flag = g_flag
            if b_value != -1:  # b_value == -1 if no change
                FrameProcessor.blur_value = b_value
            print("BEFORE DRAW_DETECTIONS")
            cropped = draw_detections(cropped, frame_processor, detections,
                                    output_transform, m_flag, emotion_detector)
            print("AFTER DRAW_DETECTIONS")
            # frame[points[1]:points[3]+400, points[0]:points[2]+400] = roi
            for x,y in np.ndindex(cropped.shape[:2]):
                if (points[1] < y and y < points[3]+400) and (points[0] < x and x < points[2]+400):
                    frame[x][y] = cropped[x][y]

            frame_num += 1
            if video_writer.isOpened() and (args.output_limit <= 0 or
                                            frame_num <= args.output_limit):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Face recognition demo', frame)
                key = cv2.waitKey(1)
                if key in {ord('q'), ord('Q'), 27}:  # 'q' 키 또는 ESC를 누르면 종료
                    break
                presenter.handleKey(key)
        print("GOT HERE TO THE END OF THE FUNCTION")


    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
