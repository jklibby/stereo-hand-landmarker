import numpy as np
import cv2 as cv
from multiprocessing import Process, get_context
import sys
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox
from collections import OrderedDict

from.test_app import normalized_to_pixel_coordinates, draw_landmarks

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
Connections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode

def _create_checkerboard_detector(cam_ids=[]):
    caps = [cv.VideoCapture(cam_id) for cam_id in cam_ids]
    WIDTH = 1920
    HEIGHT = 1080

    for cap in caps:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Something went wrong..")
        exit()
    
    while True:
        capture_frame = lambda cap: cap.read()
        accumulator = list(map(capture_frame, caps))
        result = list()
        for ret, frame in accumulator:
            if not ret:
                print("Something went wrong")
                exit()
            
            # detect and draw checkboards
            start = datetime.now()
            ret, corners = cv.findChessboardCorners(frame, (7, 7))
            print(datetime.now() - start)
            if ret:
                frame = cv.drawChessboardCorners(frame, (7, 7), corners, True)
            result.append(frame)

        for i in range(len(result)):
            cv.imshow("Frame {}".format(i), result[i])
        cv.waitKey(1)


def _create_handlandmarker(cam_ids=[]):
    caps = [cv.VideoCapture(cam_id) for cam_id in cam_ids]
    WIDTH = 1920
    HEIGHT = 1080

    for cap in caps:
        if not cap.isOpened():
            print("Something went wrong..")
            exit()
        
        cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model\handlandmarker_models\hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            capture_frame = lambda cap: cap.read()
            accumulator = list(map(capture_frame, caps))
            result = list()
            for ret, frame in accumulator:
                if not ret:
                    print("Something went wrong")
                    exit()
                
                # detect and draw checkboards
                start = datetime.now()
                ret, landmarks = _detect_landmarks(landmarker, frame)
                print(datetime.now() - start)
                if ret:
                    frame = draw_landmarks(frame, landmarks)
                result.append(frame)

            for i in range(len(result)):
                cv.imshow("Frame {}".format(i), result[i])
            cv.waitKey(1)


def _detect_landmarks(landmarker, frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(mp_image)
    if len(result.hand_landmarks) < 1:
        return False, list()
    
    landmarks = [normalized_to_pixel_coordinates(landmark.x, landmark.y, mp_image.width, mp_image.height) for landmark in result.hand_landmarks[0]]
    return True, landmarks

    



class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ctx = get_context('spawn')

        self.processes: OrderedDict[str, Process] = OrderedDict()
        self.toggle_keypoint_angle_stream = False

    def initUI(self):
        # Set up the main window
        self.setWindowTitle('Stereo Hand Landmarker')
        self.setGeometry(200, 200, 300, 200)

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create three buttons
        self.detect_landmarks = QPushButton('Detect Landmarks', self)
        self.button2 = QPushButton('Terminate App', self)


        # Connect buttons to their respective functions
        self.detect_landmarks.clicked.connect(self.on_detect_landmarks)
        self.button2.clicked.connect(self.terminate_app)
        

        # Add buttons to the layout
        layout.addWidget(self.detect_landmarks)
        layout.addWidget(self.button2)
        

        # Set the layout for the main window
        self.setLayout(layout)

    # Define the actions for each button
    def on_detect_landmarks(self):
        """
            Starts a new processes to detect landmarks from cameras
        """
        # check is existing processes are still active
        if self.processes:
            for k, process in self.processes.items():
                if process.is_alive():
                    print("Process for {} is active".format(k))
                    return
        
        
        # create process from context for single camera landmark detection
        self.processes['cb_0'] = self.ctx.Process(target=_create_handlandmarker, args=([0, 1],))
        # self.processes['cb_1'] = self.ctx.Process(target=_create_hand_detector, args=(1,))
        # self.processes['stereo_landmarker'] = self.ctx.Process(target=_create_stereo_landmarker)
        # self.processes['stereo_landmarker_visualizer'] = self.ctx.Process(target=_create_stereo_hsv)

        for process in self.processes.values():
            process.start()


    def terminate_app(self):
        QMessageBox.critical(self, 'Terminate App', 'You are closing Stereo Landmarker!',)
        for process in self.processes.values():
            process.terminate()
        self.processes = OrderedDict()


# Main application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
