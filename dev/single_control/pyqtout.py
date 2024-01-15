import cv2
import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class VideoThread(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(VideoThread, self).__init__(parent)
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(-1)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while self.running:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                self.update_signal.emit(qImg)
            else:
                QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
                print("Cannot read frame.")
                break

        cap.release()
        print("Thread end.")

class VideoPlayerApp(QtWidgets.QWidget):
    def __init__(self):
        super(VideoPlayerApp, self).__init__()

        self.video_thread = VideoThread(self)
        self.video_thread.update_signal.connect(self.update_image)

        self.label = QtWidgets.QLabel()

        self.btn_start = QtWidgets.QPushButton("Camera On")
        self.btn_stop = QtWidgets.QPushButton("Camera Off")

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.label)
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_stop)

        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)

    def update_image(self, qImg):
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)

    def start_video(self):
        self.video_thread.running = True
        self.video_thread.start()
        print("Camera On")
        # 이 스레드가 동작할 때 두 번째 스레드(=삭제 전 if문 안의 동작들)에는 들어오지 않는 모습을 보였다. 
        # 스레드 제어에 신경쓸 것. 
        blur_value = 20
        blur = cv2.GaussianBlur(self, (51, 51), blur_value)

    def stop_video(self):
        self.video_thread.running = False
        print("Camera Off")

    def closeEvent(self, event):
        self.stop_video()
        print("Exit")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = VideoPlayerApp()
    win.show()
    sys.exit(app.exec_())
