from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QVideoWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.video_widget = QVideoWidget(self.centralwidget)
        self.video_widget.setGeometry(QtCore.QRect(0, 0, 800, 480))
        self.video_widget.setObjectName("video_widget")

        self.play_button = QtWidgets.QPushButton(self.centralwidget)
        self.play_button.setGeometry(QtCore.QRect(350, 500, 100, 30))
        self.play_button.setObjectName("play_button")
        self.play_button.setText("재생")

        MainWindow.setCentralWidget(self.centralwidget)

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)

        self.play_button.clicked.connect(self.toggle_play)

        self.camera = QCamera(QCameraInfo.defaultCamera())
        self.camera.setViewfinder(self.video_widget)
        self.camera.start()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Hand", "손 모션 face 블러처리 솔루션"))

    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.camera.stop()
            video_url = QtCore.QUrl.fromLocalFile('동영상_파일_경로.mp4')  # 실제 동영상 파일 경로로 교체해주세요
            content = QMediaContent(video_url)
            self.media_player.setMedia(content)
            self.media_player.play()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())