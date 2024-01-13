"""
GUI-equipped application
"""

import sys
import os
import logging as log
from pathlib import Path
import numpy as np

from face_identifier import FaceIdentifier
from frame_processor import FrameProcessor
from face_drawer import FaceDrawer
from emotion_detector import EmotionDetector
from gesture_detector import GestureDetector
from person_detector import PersonDetector

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

import cv2

sys.path.append(str(Path(__file__).resolve().parents[0] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[0]
                    / 'common/python/openvino/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']


def draw_detections(frame, frame_processor, detections,
                    output_transform, blur_mode, emotion_detector):
    """
    Draw detections.
    """
    size = frame.shape[:2]
    frame = output_transform.resize(frame)

    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin,
                                                         xmax, ymax])

        face_block = frame[ymin:ymax, xmin:xmax]

        if identity.id == FaceIdentifier.UNKNOWN_ID:
            if blur_mode == "DEFAULT":
                # 기본값으로 얼굴 영역을 가려버린다
                frame[ymin:ymax, xmin:xmax] = cv2.GaussianBlur(
                    face_block, (51, 51), FrameProcessor.blur_value)
            elif blur_mode == "CHANGE":
                # 감정 이미지를 얼굴 영역의 높이에 맞추어 크기를 조절한다
                emotion = emotion_detector.detect_emotion(face_block)
                bgr_image = cv2.imread(emotion, cv2.IMREAD_UNCHANGED)
                emotion_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
                frame = FaceDrawer.draw_face(frame, xmin, ymin, xmax, ymax,
                                             text, landmarks, output_transform,
                                             emotion_image)
            else:  # if blur_mode == "NONE"
                pass

        # Uncomment to display text on image, 영상에 텍스트를 출력하기 위해서는 아래 주석을 푸시오
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


class Thread(QThread):
    """
    Thread class
    """
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True

    def run(self):
        """
        Start execution of thread
        """
        args = None

        cap = open_images_capture("/dev/video0", False)

        frame_processor = FrameProcessor(args)  # 영상 프레임 처리기
        gesture_detector = GestureDetector("intel/models/model.h5")
        emotion_detector = EmotionDetector("intel/emotions-recognition-\
                                           retail-0003/FP16/emotions-recognition\
                                           -retail-0003.xml")  # 얼굴 감정 인식 모델
        person_detector = PersonDetector("intel/models/ssdlite_mobilenet\
                                         _v2_fp16.xml")  # 사람 인식 모델

        frame_num = 0
        metrics = PerformanceMetrics()
        presenter = None
        output_transform = None
        input_crop = None

        crop_size = (0, 0)
        if crop_size[0] > 0 and crop_size[1] > 0:
            input_crop = np.array(crop_size)
        elif not (crop_size[0] == 0 and crop_size[1] == 0):
            raise ValueError('Both crop height and width should be positive')
        video_writer = cv2.VideoWriter()

        seq = []
        action_seq = []

        output_resolution = None
        utilization_monitors = ''
        output = None
        output_limit = 1000

        # *** 손동작 인식으로 값이 변하는 제어 변수 ***
        m_flag = "DEFAULT"  # 기본값으로 얼굴 blurring 활성화

        while self.status:
            frame = cap.read()  # 영상 입력에서 하나의 프레임을 읽어온다
            if frame is None:
                if frame_num == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if input_crop is not None:
                frame = center_crop(frame, input_crop)
            if frame_num == 0:
                output_transform = \
                    OutputTransform(frame.shape[:2], output_resolution)
                if output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                presenter = \
                    monitors.Presenter(utilization_monitors, 55,
                                    (round(output_resolution[0] / 4),
                                        round(output_resolution[1] / 8)))
                if output and not video_writer.open(
                    output, cv2.VideoWriter_fourcc(*'MJPG'),
                        cap.fps(), output_resolution):
                    raise RuntimeError("Can't open video writer")

            boxes = person_detector.detect_people(frame)  # boxes is list

            for label, score, box in boxes:
                x2 = box[0] + box[2]
                y2 = box[1] + box[3]
                roi = frame[box[1]:y2, box[0]:x2]

                # Bounding boxes around each person ROI
                # cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2),
                #               color=(255, 0, 0), thickness=3)

                detections = frame_processor.process(roi)
                presenter.drawGraphs(roi)
                # 인식된 얼굴, 손동작, 얼굴 감정에 따라 얼굴에 영상처리를 한다
                # 만약 등록된 사용자가 영상 속에 있다면 img_roi로 해당 사용자의 얼굴과 가슴 영역을 돌려준다
                g_flag, b_value = gesture_detector.detect_gesture(roi, seq,
                                                                  action_seq)

                if g_flag != "":  # g_flag == "" if no change
                    m_flag = g_flag
                if b_value != -1:  # b_value == -1 if no change
                    FrameProcessor.blur_value = b_value

                roi = draw_detections(roi, frame_processor,
                                      detections, output_transform,
                                      m_flag, emotion_detector)

                frame[box[1]:y2, box[0]:x2] = roi

            frame_num += 1
            if video_writer.isOpened() and (output_limit <= 0 or
                                            frame_num <= output_limit):
                video_writer.write(frame)

            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to fit the GUI
            h, w, ch = color_frame.shape
            target_height = self.parent().label.height()
            target_width = int((target_height / h) * w)
            resized_frame = cv2.resize(color_frame,
                                       (target_width, target_height))

            # Creating QImage
            h, w, ch = resized_frame.shape
            img = QImage(resized_frame.data, w, h,
                         ch * w, QImage.Format_RGB888)

            # Emit signal
            self.updateFrame.emit(img)

        metrics.log_total()
        for rep in presenter.reportMeans():
            log.info(rep)

    def stop(self):
        """
        Stop thread
        """
        self.status = True
        # 이 상태를 False로 바꾸면 stop 버튼을 눌렀을 때 바로 창이 꺼진다.
        self.wait()


class Window(QMainWindow):
    """
    Window class
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NuGaBa")
        self.setGeometry(0, 0, 1920, 1080)  # or 2500 * 1500

        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)

        self.thread = Thread(self)
        self.thread.finished.connect(self.close)
        self.thread.updateFrame.connect(self.setImage)
        
        self.btn_select = QPushButton("Select Face")
        self.btn_start = QPushButton("Streaming")
        self.btn_stop = QPushButton("Stop")
        self.btn_save = QPushButton("Save")
        self.btn_stop.setEnabled(False)

        self.group_controls = QGroupBox(" ")
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.btn_select)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_save)
        self.group_controls.setLayout(controls_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.group_controls)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.btn_select.clicked.connect(self.selectFace)
        self.btn_start.clicked.connect(self.startVideo)
        self.btn_stop.clicked.connect(self.stopVideo)
        self.btn_save.clicked.connect(self.saveVideo)

    @Slot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

    def selectFace(self):
        options = QFileDialog.Options()
        options != QFileDialog.ReadOnly
        file_dialog = QFileDialog(self)
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Image Files (*.jpg)")
        file_paths, _ = file_dialog.getOpenFileNames()
        if file_paths:
            print("Selected Image Files : ", file_paths)
            self.saveImages(file_paths)

    def saveImages(self, file_paths):
        output_folder = "./"
        os.makedirs(output_folder, exist_ok=True)
        for file_path in file_paths:
            img = cv2.imread(file_path)
            output_path = os.path.join(output_folder,
                                       os.path.basename(file_path))
            cv2.imwrite(output_path, img)
            print(f"Image saved to {output_path}")

    def startVideo(self):
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stopVideo(self):
        self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def saveVideo(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Saving Video", "",
                                                   "Video Files (*.avi *.mp4)")

        if file_path:
            # VideoWriter 초기화
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (XVID는 AVI 포맷 지원)
            fps = 30.0  # 초당 프레임 수
            # cv2.VideoWriter의 매개변수: 파일명, 코덱, FPS, 해상도 설정
            video_writer = cv2.VideoWriter(file_path, fourcc, fps, (640, 480))

            # 비디오 저장 로직
            # 캡처된 프레임을 처음으로 초기화
            self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = self.thread.cap.read()
                if not ret:
                    break

                # 비디오 프레임 저장
                video_writer.write(frame)

            # VideoWriter 닫기
            video_writer.release()

            print(f"Video Saved to {file_path}")


if __name__ == "__main__":
    app = QApplication()
    window = Window()
    window.show()
    # window.showFullScreen()  # Show in full screen
    sys.exit(app.exec())
