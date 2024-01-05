# To access command line arguments
import sys

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

# Subclass QMainWindow to create a custom main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        button = QPushButton("Press Me!")
        # Set fixed size of the window
        self.setFixedSize(QSize(400, 300)) # QSize takes width and height parameters
        # Set the central widget of the Window
        self.setCentralWidget(button)


# Only ONE QApplication instance per application (contains event loop)
# Pass sys.argv to permit command line arguments for the application
app = QApplication(sys.argv)
# app = QApplication([]) # For use without command line arguments

# Qt widget as window
window = MainWindow()
window.show() # windows hidden by default

app.exec() # Start event loop
