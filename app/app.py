import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox
import numpy as np
from multiprocessing import get_context, Process
from typing import Dict
from collections import OrderedDict

from model import RedisLandmarker, StereoLandmarker, KeyPointAngleRecorder
from visualization import HandStreamingVisualizer, HandlandmarkVideoStream


def _create_rlh(channel, cam_id):
    rlh = RedisLandmarker(channel, cam_id)
    rlh.run()

def _create_hsv(channel, cam_id):
    hsv = HandlandmarkVideoStream(cam_id, channel)
    hsv.consume()

def _create_stereo_landmarker():
    sl = StereoLandmarker()
    sl.run()

def _create_angle_recorder():
    recorder = KeyPointAngleRecorder("channel_points_3d")
    while True:
        ret = recorder.run()
        if not ret:
            break

def _create_stereo_hsv():
    stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")
    viz = HandStreamingVisualizer(cams=2, R=[stereo_calibration["R"]], T=[stereo_calibration["T"]])
    viz.consume()

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
        self.button3 = QPushButton('Stream keypoint angles', self)

        # Connect buttons to their respective functions
        self.detect_landmarks.clicked.connect(self.on_detect_landmarks)
        self.button2.clicked.connect(self.terminate_app)
        self.button3.clicked.connect(self.on_button3_click)

        # Add buttons to the layout
        layout.addWidget(self.detect_landmarks)
        layout.addWidget(self.button3)
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
        self.processes['cam_0_landmarker'] = self.ctx.Process(target=_create_rlh, args=("channel_cam_0", 0))
        self.processes['cam_0_visualizer'] = self.ctx.Process(target=_create_hsv, args=("channel_cam_0", 0))

        self.processes['cam_1_landmarker'] = self.ctx.Process(target=_create_rlh, args=("channel_cam_1", 1))
        self.processes['cam_1_visualizer'] = self.ctx.Process(target=_create_hsv, args=("channel_cam_1", 1))

        # create process from context for stereo landmarker
        self.processes['stereo_landmarker'] = self.ctx.Process(target=_create_stereo_landmarker)
        self.processes['stereo_landmarker_visualizer'] = self.ctx.Process(target=_create_stereo_hsv)

        for process in self.processes.values():
            process.start()


    def terminate_app(self):
        QMessageBox.critical(self, 'Terminate App', 'You are closing Stereo Landmarker!',)
        for process in self.processes.values():
            process.terminate()
        self.processes = OrderedDict()

    def on_button3_click(self):
        self.processes['key_point_angle_recorder'] = self.ctx.Process(target=_create_angle_recorder)
        self.processes['key_point_angle_recorder'].start()

# Main application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
