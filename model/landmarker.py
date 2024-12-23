from datetime import datetime, timedelta
import numpy as np
import cv2 as cv
import mediapipe as mp
from typing import *
from collections import namedtuple
from parallelism import RedisEncoder, RedisStream, SyncRedisProducer, SyncRedisConsumer
from projection import project3d


RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
Connections = mp.tasks.vision.HandLandmarksConnections

AngleKeyPointConnections = namedtuple('AngleKeyPointConnections', ['start', 'end'])
keypoint_connections = [
    # thumb
    (AngleKeyPointConnections(start=0, end=1), AngleKeyPointConnections(start=1, end=2)),
    (AngleKeyPointConnections(start=1, end=2), AngleKeyPointConnections(start=2, end=3)),
    (AngleKeyPointConnections(start=2, end=3), AngleKeyPointConnections(start=3, end=4)),

    # index
    (AngleKeyPointConnections(start=1, end=5), AngleKeyPointConnections(start=5, end=6)),
    (AngleKeyPointConnections(start=5, end=6), AngleKeyPointConnections(start=6, end=7)),
    (AngleKeyPointConnections(start=6, end=7), AngleKeyPointConnections(start=7, end=8)),

    # middle
    (AngleKeyPointConnections(start=0, end=9), AngleKeyPointConnections(start=9, end=10)),
    (AngleKeyPointConnections(start=9, end=10), AngleKeyPointConnections(start=10, end=11)),
    (AngleKeyPointConnections(start=10, end=11), AngleKeyPointConnections(start=11, end=12)),

    # ring
    (AngleKeyPointConnections(start=0, end=13), AngleKeyPointConnections(start=13, end=14)),
    (AngleKeyPointConnections(start=13, end=14), AngleKeyPointConnections(start=14, end=15)),
    (AngleKeyPointConnections(start=14, end=15), AngleKeyPointConnections(start=15, end=16)),

    # pinky
    (AngleKeyPointConnections(start=0, end=17), AngleKeyPointConnections(start=17, end=18)),
    (AngleKeyPointConnections(start=17, end=18), AngleKeyPointConnections(start=18, end=19)),
    (AngleKeyPointConnections(start=18, end=19), AngleKeyPointConnections(start=19, end=20)),
]

def configure_mp_options(model_path: str, running_mode=RunningMode.LIVE_STREAM, result_callback=lambda x, y, z: print("Default:", x)):
    # configure pose landmarker
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=running_mode,
    result_callback=result_callback
    )
    return options


class RedisHandlandMarkerResult(RedisEncoder):
    """
    A class representing the result of a hand landmarker. 
    Contains interfaces for creating redis data objects. 

    Attributes:
        cam_id (int): The camera ID from which the result is captured.
        timestamp (int): The timestamp of the result. Default is 0.
        result (HandLandmarkerResult): The result of the hand landmark detection. Default is None.
        terminate (bool): A flag indicating whether to terminate the process. Default is False.
    """
    def __init__(self, cam_id:int, timestamp:int = 0, result:HandLandmarkerResult = None, terminate:bool = False) -> None:
        self.result = result
        self.timestamp = timestamp
        self.terminate = terminate
        self.cam_id = cam_id

class Landmarker3D(RedisEncoder):
    """
    A class representing the 3D points of a landmarker. 
    Contains interfaces for creating redis data objects. 

    Attributes:
        points (list): A list of 3D points representing landmarks.
        terminate (bool): A flag indicating whether to terminate the process. Default is False.
    """
    def __init__(self, points, terminate=False) -> None:
        super().__init__()
        self.points = points
        self.terminate = terminate

class KeyPointAngles(RedisEncoder):
    def __init__(self, timestamp, thetas, rotation_axes) -> None:
        super().__init__()
        self.timestamp = timestamp
        self.thetas = thetas
        self.rotation_axes = rotation_axes

class Landmarker():
    """
    A class representing a landmarker that detects hand landmarks from a video stream using Mediapipe.

    Attributes:
        model_path (str): The path to the model used for hand landmark detection.
        cap (cv.VideoCapture): The video capture object from the camera.
        cam_id (int): The camera ID used for video capture.
        remap_x (numpy.ndarray): The x-coordinate remap for image rectification.
        remap_y (numpy.ndarray): The y-coordinate remap for image rectification.
    """
    def __init__(self, cam_id, dir="calibration") -> None:
        self.model_path = "model/handlandmarker_models/hand_landmarker.task"
        self.cap: cv.VideoCapture = cv.VideoCapture(cam_id)
        self.cam_id = cam_id
        print(self.cam_id)
        
        options = configure_mp_options(self.model_path, result_callback=self.handle_result)
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # load rectification maps
        self.remap_x = None
        self.remap_y = None
        remaps = np.load("camera_extrinsics/stereo_rectification/stereo_rectification_maps.npz")
        if self.cam_id == 0:
            self.remap_x, self.remap_y = remaps['left_map_x'], remaps['left_map_y']
        elif self.cam_id == 1:
            self.remap_x, self.remap_y = remaps['right_map_x'], remaps['right_map_y']

    def rectify_image(self, frame) -> np.ndarray:
        """
        Rectifies the given image frame using the loaded remap matrices.

        Args:
            frame (numpy.ndarray): The image frame to be rectified.

        Returns:
            numpy.ndarray: The rectified image frame.
        """
        r_frame = cv.remap(frame, self.remap_x, self.remap_y, cv.INTER_LANCZOS4)
        # cv.imwrite("recitfied_frame_{}.png".format(self.cam_id), r_frame)
        return r_frame

    def handle_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """
        Handles the result from the hand landmarker To be implemented by Child Class.

        Args:
            result (HandLandmarkerResult): The result from the hand landmark detection.
            output_image (mp.Image): The output image containing the landmarks.
            timestamp_ms (int): The timestamp at which the result was obtained.
        """
        pass
        
    def detect_landmarks(self) -> bool:
        """
        Detects hand landmarks from the video stream.

        Raises:
            Exception: If the video capture cannot be opened.

        Returns:
            bool: Always returns False when the detection process is finished.
        """
        if not self.cap.isOpened():
            self.cap.release()
            raise Exception("Could not capture video")
        start = datetime.now()
        print("Started Detecting...", start)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = (datetime.now() - start).total_seconds() * 1e3
            mp_timestamp = mp.Timestamp.from_seconds(timestamp).value

            # recitfy frame
            rectified_frame = self.rectify_image(frame)

            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rectified_frame)
            self.landmarker.detect_async(mp_frame, mp_timestamp)

            # cv.imshow("Cam: {}".format(self.cam_id), frame)
            # if timestamp > 2*1e4:
            #     break
        self.cap.release()
        cv.destroyAllWindows()
        return False


class RedisLandmarker(Landmarker, SyncRedisProducer):
    """
    A class that extends the Landmarker and RedisProducer classes, combining hand landmark detection with Redis messaging.

    Attributes:
        channel (str): The Redis channel for sending results.
        cam_id (int): The camera ID used for video capture.
    """
    def __init__(self, channel: str, cam_id: int) -> None:
        Landmarker.__init__(self, cam_id)
        SyncRedisProducer.__init__(self, channel)
    
    def handle_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """
        Handles the result from the hand landmarker, producing the result to the Redis channel.

        Args:
            result (HandLandmarkerResult): The result from hand landmark detection.
            output_image (mp.Image): The image with landmarks overlayed.
            timestamp_ms (int): The timestamp when the result was produced.
        """
        if len(result.hand_landmarks) < 1:
            return
        
        print("Detect in {}".format(self.cam_id))
        extended_result = RedisHandlandMarkerResult(self.cam_id, timestamp_ms, result)
        self.produce(extended_result)

        
    
    def run(self) -> None:
        """
        Runs the landmark detection process and produces a termination message once complete.
        """
        self.detect_landmarks()
        terminate = RedisHandlandMarkerResult(self.cam_id, terminate=True)
        self.produce(terminate)


class StereoLandmarker(SyncRedisConsumer, SyncRedisProducer):
    """
    A class that combines RedisConsumer and SyncRedisProducer, used for consuming hand landmark detection results from two cameras and producing 3D points.
    """
    def __init__(self) -> None:
        SyncRedisConsumer.__init__(self, ["channel_cam_0", "channel_cam_1"])
        SyncRedisProducer.__init__(self, "channel_points_3d")

    
    def consume(self, message_objs: List[RedisHandlandMarkerResult]):
        """
        Consumes messages containing hand landmark detection results, checks for landmarks in both frames, and produces 3D points.

        Args:
            message_objs (List[RedisHandlandMarkerResult]): A list of hand landmark detection results from both cameras.

        Returns:
            bool: Returns True to keep consuming if landmarks are not detected in both frames, False to stop consuming if terminated.
        """
        # check if landmark have been detected in both frames
        
        if not(message_objs[0].result.hand_landmarks and message_objs[1].result.hand_landmarks):
            # keep listening
            print("exit as no landmark detected")
            return True
        print("Landmark detected...", message_objs[0].timestamp, message_objs[1].timestamp)

        # get points for each camera
        landmarks = landmarks_to_numpy(message_objs)
        # get 3d points for each camera
        points3d = project3d(landmarks[:, 0], landmarks[:, 1], "calibration")
        l3d = Landmarker3D(points3d.tolist())
        # l3d = Landmarker3D(world_landmarks(message_objs))
        # print("+"*30, l3d.shape)
        # push them into a pub/sub queue
        self.produce(l3d)
    
    def run(self) -> None:
        """
        Starts listening to messages on multiple channels, converting and consuming them.
        """
        while True:
            for m1, m2 in zip(self.pubsubs[0].listen(), self.pubsubs[1].listen()):
                transformed_messages = self.convert_messages([m1, m2])
                keep_listening = self.consume(transformed_messages)
                if not keep_listening:
                    break
            

class KeyPointAngleRecorder(SyncRedisConsumer, RedisStream):
    def __init__(self, channel: str | List[str]):
        SyncRedisConsumer.__init__(self, channel)
        RedisStream.__init__(self)


    def run(self):
        # get 3D key points from the redis channel
        start_time = None
        for message in self.pubsub.listen():
            if start_time:
                timestamp = datetime.now() - start_time
            else:
                timestamp = timedelta(seconds=0, microseconds=0)
                start_time = datetime.now()
            if timestamp > timedelta(seconds=30):
                break
            message = self.convert_messages(message)
            thetas, rotation_axes = self.extract_rotation_axis_angle_v2(message.points)
             # add them to redis stream, create a new stream everytime run is invoked
            self.produce(start_time.strftime('%Y-%m-%d-%s'), KeyPointAngles(timestamp=timestamp, thetas=thetas, rotation_axes=rotation_axes))
        return False
    
    def extract_rotation_axis_angle(self, points: List):
        eps = 1e-10
        thetas = list()
        rotation_axes = list()
        for connection in Connections.HAND_CONNECTIONS:
            # calculate angles and rotation axis between keypoints
            landmark_1, landmark_2 = points[connection.start], points[connection.end]
            theta = float(np.arccos(np.clip(np.dot(landmark_1, landmark_2) / (np.linalg.norm(landmark_1) * np.linalg.norm(landmark_2)), -1, 1)))
            rotation_axis = np.cross(landmark_1, landmark_2)
            axis_norm = np.linalg.norm(rotation_axis) + eps
            rotation_axis = rotation_axis / axis_norm
            thetas.append(theta)
            rotation_axes.append(rotation_axis.tolist())
        return thetas, rotation_axes
    
    def extract_rotation_axis_angle_v2(self, points: List):
        thetas = list()
        rotation_axes = list()
        for start_connection, end_connection in keypoint_connections:
            # calculate angles and rotation axis between keypoints
            vec_1 = np.array(points[start_connection.start]) - np.array(points[start_connection.end])
            vec_2 = np.array(points[end_connection.start]) - np.array(points[end_connection.end])
            theta = float(np.arccos(np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))))
            rotation_axis = np.cross(vec_1, vec_2)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            thetas.append(theta)
            rotation_axes.append(rotation_axis.tolist())
        return thetas, rotation_axes



def world_landmarks(message_objs: List[RedisHandlandMarkerResult]):
    all_landmarks = list()
    for landmark_id in range(21):
        curr_landmark = list()
        for message in message_objs:
            landmark = message.result.hand_world_landmarks[0][landmark_id]
            curr_landmark.append((landmark.x, landmark.y, landmark.z))
        all_landmarks.append(curr_landmark)
    as_numpy = np.array(all_landmarks)
    return as_numpy

def landmarks_to_numpy(message_objs: List[RedisHandlandMarkerResult]):
    # 21 landmarks - mediapipe
    all_landmarks = list()
    for landmark_id in range(21):
        curr_landmark = list()
        for message in message_objs:
            landmark = message.result.hand_landmarks[0][landmark_id]
            normalized_coordinates = normalized_to_pixel_coordinates(landmark.x, landmark.y, 1920, 1080)
            curr_landmark.append(normalized_coordinates)
        all_landmarks.append(curr_landmark)
    as_numpy = np.array(all_landmarks)
    return as_numpy

def check_for_landmarks(message: RedisHandlandMarkerResult) -> bool:
    landmarks = message.result.hand_landmarks
    return len(landmarks) > 0


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
  return x_px, y_px