import sys
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPalette
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import face_recognition
import imutils
import pickle
import cv2
import datetime
import time
import threading
import copy
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pymongo import MongoClient
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QGroupBox, QListWidget, QListWidgetItem


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    appendUser = pyqtSignal(str)
    setName = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Koneksi ke MongoDB
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["face_detection"]
            self.absensi_collection = self.db["absensi"]
            self.employees_collection = self.db["employee"]
        except Exception as e:
            print(f"Database connection error: {e}")
        
    def run(self):
        currentname = "unknown"
        encodingsP = "encodings.pickle"
        cascade = "haarcascade_frontalface_default.xml"

        hand_detector = HandDetector(detectionCon=0.5, maxHands=1)

        print("[INFO] loading encodings + face detector...")
        data = pickle.loads(open(encodingsP, "rb").read())
        detector = cv2.CascadeClassifier(cascade)

        print("[INFO] starting video stream...")

        wCam, hCam = 1920, 1080
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(4, hCam)

        prev_frame_time = 0
        recognized_faces = {}

        while True:
            if not cap.isOpened():
                print("Camera not available")
                continue

            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            imgCopy = copy.deepcopy(img)
            try:
                imgCopy = hand_detector.findHands(imgCopy, draw=True)
                hand_detector.findPosition(imgCopy)
            except Exception as e:
                pass

            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time
            cv2.putText(img, f'{fps} FPS', (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

            frame = imutils.resize(img, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                    if name != "Unknown":
                        current_time = datetime.datetime.now()
                        if name not in recognized_faces:
                            recognized_faces[name] = {"arrival": current_time, "departure": None}
                            # Ambil data bagian dari koleksi employees
                            employee_data = self.employees_collection.find_one({"name": name})
                            bagian = employee_data["bagian"] if employee_data else "Tidak diketahui"

                            # Emit untuk UI
                            self.appendUser.emit(f"{name} ({bagian}) hadir pada {current_time.strftime('%H:%M:%S')}")
                            self.setName.emit(name)

                            # Simpan ke MongoDB absensi
                            self.absensi_collection.insert_one({
                                "name": name,
                                "arrival": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                "departure": None,
                                "bagian": bagian
                            })
                        else:
                            if recognized_faces[name]["departure"] is None:
                                time_diff = (current_time - recognized_faces[name]["arrival"]).seconds
                                if time_diff > 3600:  # Misalnya setelah 3600 detik baru diizinkan mencatat kepulangan
                                    recognized_faces[name]["departure"] = current_time
                                    self.appendUser.emit(f"{name} pulang pada {current_time.strftime('%H:%M:%S')}")
                                    self.setName.emit(name)
                                    # Update MongoDB
                                    self.absensi_collection.update_one(
                                        {"name": name, "departure": None},
                                        {"$set": {"departure": current_time.strftime('%Y-%m-%d %H:%M:%S')}}
                                    )

                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(509, 521, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        self.setWindowTitle('Face Recognition Attendance System')
        self.showFullScreen()
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eef2f3, stop:1 #ffffff);
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QListWidget {
                background-color: white;
                border-radius: 10px;
                padding: 5px;
                font-size: 14px;
                color: #444444;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout()

        # Header with custom control buttons
        header_layout = QHBoxLayout()
        header_label = QLabel('Face Recognition Attendance System')
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #444444;")
        header_label.setAlignment(Qt.AlignCenter)

        # Control buttons
        self.minimize_button = QPushButton("-")
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(self.showMinimized)

        self.restore_button = QPushButton("⬜")
        self.restore_button.setFixedSize(30, 30)
        self.restore_button.clicked.connect(self.toggleRestoreMaximize)

        self.close_button = QPushButton("X")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.minimize_button)
        control_layout.addWidget(self.restore_button)
        control_layout.addWidget(self.close_button)

        header_layout.addWidget(header_label, stretch=1)
        header_layout.addLayout(control_layout)
        main_layout.addLayout(header_layout)

        # Content layout
        content_layout = QHBoxLayout()

        # Left column (video feed and timer)
        left_column = QVBoxLayout()

        # Video feed
        self.label = QLabel()
        self.label.setFixedSize(500, 375)
        self.label.setStyleSheet("background-color: #f9f9f9; border: 2px solid #cccccc; border-radius: 10px;")
        left_column.addWidget(self.label)

        # Timer
        self.label_timer = QLabel()
        self.label_timer.setAlignment(Qt.AlignCenter)
        self.label_timer.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 15px; color: #555555;")
        left_column.addWidget(self.label_timer)

        content_layout.addLayout(left_column)

        # Right column (recognized person and attendance list)
        right_column = QVBoxLayout()

        # Currently recognized person
        recognized_group = QGroupBox("Currently Recognized")
        recognized_layout = QVBoxLayout()
        self.label_name = QLabel()
        self.label_name.setStyleSheet("font-size: 20px; font-weight: bold; color: #444444;")
        recognized_layout.addWidget(self.label_name)
        recognized_group.setLayout(recognized_layout)
        right_column.addWidget(recognized_group)

        # Attendance list
        attendance_group = QGroupBox("Recent Attendance")
        attendance_layout = QVBoxLayout()
        self.listWidget = QListWidget()
        attendance_layout.addWidget(self.listWidget)
        attendance_group.setLayout(attendance_layout)
        right_column.addWidget(attendance_group)

        content_layout.addLayout(right_column)
        main_layout.addLayout(content_layout)

        # Set the main layout
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Start thread to handle camera feed and face recognition
        self.thread = Thread()
        self.thread.changePixmap.connect(self.setImage)
        self.thread.appendUser.connect(self.appendUser)
        self.thread.setName.connect(self.setName)
        self.thread.start()

        # Timer to update clock
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        self.timer.start(1000)

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def updateTime(self):
        dt_now_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.label_timer.setText(dt_now_str)

    def appendUser(self, user):
        self.listWidget.addItem(user)

    def setName(self, name):
        self.label_name.setText(name)

    # Menambahkan toggleRestoreMaximize
    def toggleRestoreMaximize(self):
        if self.isMaximized():
            self.showNormal()  # Beralih ke mode normal
        else:
            self.showMaximized()  # Beralih ke mode maximize

app = QtWidgets.QApplication(sys.argv)
window = Ui()
window.show()
sys.exit(app.exec_())

