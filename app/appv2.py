import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLineEdit, QFormLayout
from PyQt5.QtGui import QIntValidator
import numpy as np
from pathlib import Path
from multiprocessing import get_context, Process, Manager
from multiprocessing.managers import BaseManager
from visualization import HandStreamingVisualizer, StreamingVisualizer
from parallelism import RedisProducer, RedisEncoder
from parallelism.redis import RedisStreamConsumer
from estimation.gpr import fit_gpr
from typing import *
from collections import OrderedDict, deque, namedtuple, defaultdict
import cv2 as cv
from datetime import datetime, timedelta
from branca.colormap import linear
from model import StereoLandmarker, KeyPointAngleRecorder

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
Connections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode

PATTERN = (7, 7)

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or np.isclose(0, value)) and (value < 1 or
                                                      np.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return -1, -1
  x_px = min(np.floor(normalized_x * image_width), image_width - 1)
  y_px = min(np.floor(normalized_y * image_height), image_height - 1)
  return float(x_px), float(y_px)

def _create_checkerboard_detector(cam_id):
    cap = cv.VideoCapture(cam_id)
    WIDTH = 1920
    HEIGHT = 1080


    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Something went wrong..")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Something went wrong...")
            exit()
        ret, corners = cv.findChessboardCorners(frame, PATTERN, None)
        print("Found corners...")
        if ret:
            frame = cv.drawChessboardCorners(frame, PATTERN, corners, ret)

        cv.imshow("Frame {}".format(cam_id), frame)
        cv.waitKey(1)


def _create_stereo_landmarker():
    sl = StereoLandmarker()
    sl.run()

def _create_stereo_hsv():
    stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")
    viz = HandStreamingVisualizer(cams=2, R=[stereo_calibration["R"]], T=[stereo_calibration["T"]])
    viz.consume()

def _create_hand_detector(cam_id, init_time):
    q = deque([])
    producer = RedisProducer("channel_cam_{}".format(cam_id))
    def handle(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if len(result.hand_landmarks) > 0:
            q.append((timestamp_ms, np.array(output_image.numpy_view()), [normalized_to_pixel_coordinates(landmark.x, landmark.y, output_image.width, output_image.height) for landmark in result.hand_landmarks[0]]))
            
    remaps = np.load("camera_extrinsics/stereo_rectification/stereo_rectification_maps.npz")
    if cam_id == 0:
        remap_x, remap_y = remaps['left_map_x'], remaps['left_map_y']
    elif cam_id == 1:
        remap_x, remap_y = remaps['right_map_x'], remaps['right_map_y']

    model_path = Path("model/handlandmarker_models/hand_landmarker.task")
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle)
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv.VideoCapture(cam_id)
        WIDTH = 1920
        HEIGHT = 1080

        cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        if not cap.isOpened():
            print("Something went wrong..")
            exit()
        
        start = datetime.now()
        while True:
            ret, frame = cap.read()
            # frame = cv.remap(frame, remap_x, remap_y, cv.INTER_LANCZOS4)

            if not ret:
                print("Something went wrong while reading the next frame...")
                exit()

            curr_time = datetime.now()
            if curr_time > init_time:
                timestamp = int(curr_time.timestamp() * 1e6)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)    
                landmarker.detect_async(mp_image, timestamp)

            
            if q:
                detected_ts, frame, landmarks = q.popleft()
                item = LandmarkQueueItem(cam_id, landmarks, detected_ts)
                producer.produce(item)
                frame = draw_landmarks(frame, landmarks)
                

            cv.imshow("Frame {}".format(cam_id), frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break
        
        cap.release()
        cv.destroyAllWindows()

def color_bar_np(length):
    colormap = getattr(linear, 'viridis').scale(0, 1)
    lower_color = colormap(0)
    upper_color = colormap(1)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    lower_color_rgb = hex_to_rgb(lower_color)
    upper_color_rgb = hex_to_rgb(upper_color)
    color_variations = np.linspace(lower_color_rgb, upper_color_rgb, length)
    return color_variations

def draw_landmarks(frame, landmarks):
        landmarks = np.int32(landmarks)
        colors = color_bar_np(landmarks.shape[0])
        for idx, landmark in enumerate(landmarks):
            cv.circle(frame,landmark, radius=5, color=colors[idx], thickness=1)
        
        for idx, connection in enumerate(Connections.HAND_CONNECTIONS):
            cv.line(frame, landmarks[connection.start], landmarks[connection.end], color=colors[idx], thickness=5)
        
        return frame

def _create_angle_recorder(stream_key):
    recorder = KeyPointAngleRecorder(stream_key, "channel_points_3d")
    recorder.run()

def _replay_stream(stream_key):
    angles = RedisStreamConsumer().consume(stream_key)
    sv = StreamingVisualizer()
    sv.replay_angles(angles)

def _fit_gpr(streams):
    angles = fit_gpr(streams)
    sv = StreamingVisualizer()
    sv.replay_angles(angles)
    

        
class LandmarkQueueItem(RedisEncoder):
    def __init__(self, cam_id, result, timestamp):
       self.cam_id = cam_id
       self.result = result
       self.timestamp = timestamp

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
        form_layout = QFormLayout()

        self.cam_id_1 = QLineEdit()
        self.cam_id_1.setPlaceholderText("Camera 1 ID")
        self.cam_id_1.setValidator(QIntValidator(0, 10))
        
        self.cam_id_2 = QLineEdit()
        self.cam_id_2.setPlaceholderText("Camera 2 ID")
        self.cam_id_2.setValidator(QIntValidator(0, 10))

        self.stream_name = QLineEdit()
        self.stream_name.setPlaceholderText("Stream name to store key point angles")

        self.stream_names = QLineEdit()
        self.stream_names.setPlaceholderText("Stream names to fit GPR on")
        
        # Create three buttons
        self.detect_landmarks = QPushButton('Detect Landmarks', self)
        self.record_keypoint_angles = QPushButton('Record Keypoint angles', self)
        self.replay_keypoint_angles = QPushButton('Replay Keypoint angles', self)
        self.fit_gpr_button = QPushButton('Fit GPR', self)
        self.button2 = QPushButton('Terminate App', self)


        # Connect buttons to their respective functions
        self.detect_landmarks.clicked.connect(self.on_detect_landmarks)
        self.record_keypoint_angles.clicked.connect(self.on_record_keypoint_angles)
        self.replay_keypoint_angles.clicked.connect(self.on_replay_keypoint_angles)
        self.fit_gpr_button.clicked.connect(self._on_fit_gpr)
        self.button2.clicked.connect(self.terminate_app)
        

        # Add buttons to the layout
        form_layout.addRow("Camera ID 1", self.cam_id_1)
        form_layout.addRow("Camera ID 2", self.cam_id_2)
        form_layout.addRow("Stream Name", self.stream_name)
        form_layout.addRow("Stream Names", self.stream_names)
        layout.addLayout(form_layout)
        layout.addWidget(self.detect_landmarks)
        layout.addWidget(self.record_keypoint_angles)
        layout.addWidget(self.replay_keypoint_angles)
        layout.addWidget(self.fit_gpr_button)
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
        init_time = datetime.now() + timedelta(seconds=10)
        self.processes['cb_0'] = self.ctx.Process(target=_create_hand_detector, args=(int(self.cam_id_1.text()), init_time,))
        self.processes['cb_1'] = self.ctx.Process(target=_create_hand_detector, args=(int(self.cam_id_2.text()), init_time, ))
        self.processes['stereo_landmarker'] = self.ctx.Process(target=_create_stereo_landmarker)
        self.processes['stereo_landmarker_visualizer'] = self.ctx.Process(target=_create_stereo_hsv)

        for process in self.processes.values():
            process.start()

    def on_record_keypoint_angles(self):
        self.record_keypoint_angles.setEnabled(False)
        _create_angle_recorder(self.stream_name.text())
        self.record_keypoint_angles.setEnabled(True)
    
    def on_replay_keypoint_angles(self):
        self.replay_keypoint_angles.setEnabled(False)
        _replay_stream(self.stream_name.text())
        self.replay_keypoint_angles.setEnabled(True)
    
    def _on_fit_gpr(self):
        streams = self.stream_names.text().split(",")
        _fit_gpr(streams)
        
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
