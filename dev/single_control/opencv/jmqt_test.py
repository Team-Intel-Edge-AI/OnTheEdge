import queue
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

#from receive_frame import ReceiveFrame


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.frame_queue = queue.Queue(maxsize=30)    # 프레임이 저장될 큐
        self.startThread()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.camViewer)
        self.timer.start(30)

    def initUI(self):
        # UI 틀 만들기
        self.setWindowTitle('BTS')
        widget = QWidget()
        self.setCentralWidget(widget)
        
        vbox = QVBoxLayout(widget)
        
        pixmap = QPixmap('image/supersonic.jpg')
        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(pixmap.scaled(640, 480, aspectRatioMode=Qt.KeepAspectRatio))
        
        lbl_size = QLabel('Hi')
        lbl_size.setAlignment(Qt.AlignCenter)
        
        vbox.addWidget(self.lbl_img)
        vbox.addWidget(lbl_size)
        self.setLayout(vbox)

        self.move(1000, 600)
        self.show()

    def startThread(self):
        self.video_thread = ReceiveFrame(self.frame_queue)
        self.video_thread.start()
        
    def camViewer(self):
        if self.frame_queue:
            frame = self.frame_queue.get()
            
            # 프레임 크기 확인
            height, width, channels = frame.shape
            bytes_per_line = channels * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            if q_img:
                pixmap = QPixmap.fromImage(q_img)
                self.lbl_img.setPixmap(pixmap)
            else:
                print("Invalid Image Data")
                
    def closeThread(self, event):
        self.video_thread.quit()
        self.video_thread.wait()
        event.accept()




"""
# 서버 설정
host = '127.0.0.1'
port = 8080

class ReceiveFrame(QThread):    
    def __init__(self, frame_queue):
        super().__init__()        
        self.frame_queue = frame_queue
        
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"서버 대기 중... {self.host}:{self.port}")
        
    def __del__(self):
        self.server_socket.close()
        print("서버를 닫습니다..")
        
    def run(self):
        # 클라이언트 연결 대기
        client_socket, addr = self.server_socket.accept()
        print(f"클라이언트 연결됨: {addr}")
        
        while True:
            frame_len = struct.unpack("!I", client_socket.recv(4))[0]
            frame_data = b""
            
            while len(frame_data) < frame_len:
                chunk = client_socket.recv(min(frame_len - len(frame_data), 4096))
                if not chunk:
                    break
                frame_data += chunk
                
            frame_np = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), 1)
            self.frame_queue.put(frame_np)
"""


