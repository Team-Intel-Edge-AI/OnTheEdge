import sys
import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from PySide6.QtWidgets import QFileDialog

# ... (rest of your code)




class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to fit the GUI
            h, w, ch = color_frame.shape
            target_height = self.parent().label.height()
            target_width = int((target_height / h) * w)
            resized_frame = cv2.resize(color_frame, (target_width, target_height))

            # Creating QImage
            h, w, ch = resized_frame.shape
            img = QImage(resized_frame.data, w, h, ch * w, QImage.Format_RGB888)

            # Emit signal
            self.updateFrame.emit(img)

    def stop(self):
        self.status = False
        self.wait()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Privacy Face Filter Streaming")	# 창 타이틀
        self.setGeometry(0, 0, 1920, 1080)	# 윈도우 창 사이즈(가로 X 세로)

        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)

        self.thread = Thread(self)
        self.thread.finished.connect(self.close)
        self.thread.updateFrame.connect(self.setImage)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_load_image = QPushButton("Load Image")		# 사용자 얼굴 이미지 저장 버튼
        self.btn_stop.setEnabled(False)

        self.group_controls = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(self.btn_load_image)		# 이미지 저장 
        self.group_controls.setLayout(controls_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.group_controls)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.btn_start.clicked.connect(self.startVideo)
        self.btn_stop.clicked.connect(self.stopVideo)
        self.btn_load_image.clicked.connect(self.loadImage)	# 이미지 저장
        

    @Slot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

    def startVideo(self):
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stopVideo(self):
        self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def loadImage(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg)")
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                color_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = color_frame.shape
                target_height = self.label.height()
                target_width = int((target_height / h) * w)
                resized_frame = cv2.resize(color_frame, (target_width, target_height))
                img = QImage(resized_frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.updateFrame.emit(img)

    def saveImage(self):
        if not hasattr(self, 'loaded_image'):
            return

        save_dialog = QFileDialog()
        save_path, _ = save_dialog.getSaveFileName(self, "Save Image", "", "Images (*.jpg)")
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
