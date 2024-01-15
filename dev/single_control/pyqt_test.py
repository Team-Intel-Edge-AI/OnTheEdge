import sys

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

# 영상 출력 반복 스레드
class Thread(QThread):
    changePixmap = pyqtSignal(QImage, np.ndarray)
    recordVideo = pyqtSignal(QImage)
    working = True
    cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)  # 0번 카메라 연결
        # self.cap = cv2.VideoCapture('move_file.avi')  # 동영상 파일

        while self.working:  # 영상 가져오기 무한 반복
            ret, frame = self.cap.read()  # 카메라에서 영상 받기 (넘파이 배열)
            if not ret: break             # 영상이 없으면 (동영상 끝 등) 반복 종료
            # 제대로 1배속 하려면 여기서 동영상의 초당 프레임수의 역수만큼 기다려야 할 듯

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
            h, w, ch = rgbImage.shape     # 영상의 세로, 가로, 채널수
            bytes_per_line = ch * w

            print(type(rgbImage.data))
            print(rgbImage.data)
            print(type(rgbImage))

            cvc = QImage(
                rgbImage.data,
                w, h, bytes_per_line,
                QImage.Format_RGB888)

            self.changePixmap.emit(
                cvc.scaled(640, 480, Qt.KeepAspectRatio),
                frame)

            break

        self.cap.release()


# Qt 창
class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        self.initUI()  # UI 초기화
        self.th = Thread(self)  # 영상 출력 반복 스레드

    # UI 초기화 (영상 출력될 라벨 한 개, 시작버튼 한 개, 중지버튼 한 개)
    def initUI(self):
        loadUi('test1.ui', self)  # ui 파일 읽어오기

    # 시작 버튼 누름
    def cam_open(self):
        self.th.working = True  # 영상 읽어오기 반복 여부 참
        self.th.changePixmap.connect(self.setImage)
        self.th.start()

    # 중지 버튼 누름
    def cam_close(self):
        self.th.working = False  # 영상 읽어오기 반복 여부 거짓
        self.th.exec_()

    # 라벨에 이미지 출력하기 함수
    def setImage(self, image):
        self.label_view.setPixmap(QPixmap.fromImage(image))


# 메인: Qt창 시작
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
