import sys
import os
import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot, QCoreApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True

    def run(self):
        self.cap = cv2.VideoCapture(0)
        # 고해상도 cam에서만 사용할 수 있는 코드 
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 3840
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440) # 2560 
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
        self.cap.release()  # Release the video capture object

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NuGaBa")
        self.setGeometry(0, 0, 1920, 1080)  # or 2500 * 1500

        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setAlignment(Qt.AlignCenter)

        self.thread = Thread(self)
        # self.thread.finished.connect(self.close)
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
        output_folder = "./"  # 이 경로를 주희와 우선이 사진이 있는 폴더로 변경해야 한다.  
        os.makedirs(output_folder, exist_ok=True)
        for file_path in file_paths:
            img = cv2.imread(file_path)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
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
        self.thread.stop()
        self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Saving Video", "", "Video Files (*.avi *.mp4)")
        # mp4로 저장하면 안 됨. 현재 이 코드의 코덱은 avi 포맷만 지원한다. 

        if file_path:  
        	# VideoWriter 초기화
        	fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (H.264로 변경하고 싶음)
        	fps = 30.0  # 초당 프레임 수
        	video_writer = cv2.VideoWriter(file_path, fourcc, fps, (1920, 1080))  # 파일명, FPS, 해상도 설정
        	
        	# 비디오 저장 로직
        	# self.thread.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 캡처된 프레임을 처음으로 초기화
        	while True:
        		ret, frame = self.thread.cap.read()
        		if not ret:
        			break
        		
        		# 비디오 프레임 저장
        		video_writer.write(frame)
        	
        	# VideoWriter 닫기
        	video_writer.release()
        	print(f"Video Saved to {file_path}")
        # self.cap.release() 

if __name__ == "__main__":
    app = QApplication()
    window = Window()
    window.show()
    # window.showFullScreen()  # Show in full screen
    sys.exit(app.exec())
