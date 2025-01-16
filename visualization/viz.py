import cv2 as cv
import open3d as o3d
import numpy as np
from branca.colormap import linear
import mediapipe as mp
import threading
from typing import *
import time
from collections import defaultdict
from parallelism import SyncRedisConsumer
from model import landmarks_to_numpy
from model.landmarker import keypoint_connections
import functools

Connections = mp.tasks.vision.HandLandmarksConnections
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

DEFAULT_HAND_POINTS = np.array([
    # wrist
    [0, 0, 0],

    #thumb
    [-1, 1, 0],
    [-2, 2, 0],
    [-3, 3, 0],
    [-4, 4, 0],

    # index
    [-2, 4, 0],
    [-2, 5, 0],
    [-2, 6, 0],
    [-2, 7, 0],

    # middle
    [0, 4, 0],
    [0, 5, 0],
    [0, 6, 0],
    [0, 7, 0],

    # ring
    [2, 4, 0],
    [2, 5, 0],
    [2, 6, 0],
    [2, 7, 0],

    # pinky
    [4, 4, 0],
    [4, 5, 0],
    [4, 6, 0],
    [4, 7, 0],

])

CONNECTIONS_DAG = defaultdict(set)
for conn_1, conn_2 in keypoint_connections:
    CONNECTIONS_DAG[conn_1.start].add(conn_1.end)
    CONNECTIONS_DAG[conn_2.start].add(conn_2.end)

def skew_symmetric(k):
    kx, ky, kz = k
    return np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])

def Rodrigues_matrix(angle, axis):
    K = skew_symmetric(axis)
    theta = angle
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def load_extrinsics():
    stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")
    return [stereo_calibration["R"]], [stereo_calibration["T"]]

class Visualizer():
    """
    A class for visualizing Hand Landmarks in 3D space using Open3D.

    Attributes:
        cams (int): The number of cameras used in the calibration.
        R (list): A list of rotation matrices for each camera.
        T (list): A list of translation vectors for each camera.
        camera_positions (list): A list of calculated camera positions in 3D space.
        viz (o3d.visualization.Visualizer): The Open3D Visualizer object for rendering the scene.
    """
    def __init__(self, cams=2, R=[], T=[]):
        self.cams = cams
        self.R = R
        self.T = T

        if not (R and T):
            r, t = load_extrinsics()
            self.R = r
            self.T = t

        
        self.camera_positions = [np.array([0, 0, 0], dtype=np.float32)]
        for r, t in zip(self.R, self.T):
            t = t.flatten()
            camera_pos = -1 * t.copy()
            self.camera_positions.append(camera_pos)
        
        print("These", self.camera_positions, len(self.camera_positions))
        self.viz: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        offset = np.mean(self.camera_positions, axis=0)
        offset[2] = 5
        self.hands = HandLineset(offset)


    def display_scene(self):
        """
        creates an Open3D scene with cameras
        """
        # convert camera coordinates to point clouds

        self.viz.create_window()

        camera_pcd = o3d.geometry.PointCloud()
        camera_pcd.points = o3d.utility.Vector3dVector(self.camera_positions)
        colors = [(255, 0, 0), (0, 0, 255)]
        camera_pcd.colors =  o3d.utility.Vector3dVector(colors)

        self.viz.add_geometry(camera_pcd)
        
        for idx, cam_pos in enumerate(self.camera_positions):
            cam_line_pcd = camera_lineset(cam_pos, 3, 3, color=colors[idx])
            self.viz.add_geometry(cam_line_pcd)
        
        self.viz.add_geometry(self.hands.hand_pcd)
        self.viz.add_geometry(self.hands.hand_lineset)

    def update_scene(self, landmarks):
        updated_hands_pcd, updated_hand_lineset = self.hands.update(landmarks)
        self.viz.update_geometry(updated_hands_pcd)
        self.viz.update_geometry(updated_hand_lineset)
        self.viz.poll_events()
        self.viz.update_renderer()

def camera_lineset(center, w, h, color):
    camera_plane = np.array([center] * 4)
    scaling = np.array([
        [-1*h/2, -1*w/2, 0],
        [-1*h/2, +1*w/2, 0],
        [+1*h/2, +1*w/2, 0],
        [+1*h/2, -1*w/2, 0],
    ])
    camera_plane += scaling

    tunnel_plane = np.array([center] * 4) + scaling * 0.5
    tunnel_plane[:, 2] = -1.5 + center[2]

    tunnel_plane_2 = tunnel_plane.copy()
    tunnel_plane_2[:, 2] = -2 + center[2]

    camera_points = np.vstack([camera_plane, tunnel_plane, tunnel_plane_2])
    camera_lines = list()
    for i in range(3):
        for j in range(4):
            if j == 3:
                camera_lines.append((j + 4*i, i*4))
            else:
                camera_lines.append((j + 4*i, j + 4*i + 1))

            if i > 0:
                camera_lines.append((j+(i-1)*4, j+ i*4))
    colors = o3d.utility.Vector3dVector([color] * len(camera_lines))
    
    camera_lines = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(camera_points), lines=o3d.utility.Vector2iVector(camera_lines))
    camera_lines.colors = colors
    return camera_lines

class HandLineset():
    def __init__(self, offset, landmarks=None) -> None:
        if landmarks is None:
            self.landmarks = DEFAULT_HAND_POINTS + offset
        pcd, lineset = hand_lineset(self.landmarks)
        self.hand_pcd = pcd
        self.hand_lineset = lineset
    
    def update(self, lndmk):
        # convert landmarks to vectors
        landmarks = o3d.utility.Vector3dVector(lndmk)
        self.hand_pcd.points = landmarks
        self.hand_lineset.points = landmarks

        self.hand_lineset.rotate(self.hand_lineset.get_rotation_matrix_from_xyz((np.pi, np.pi, 0)))
        self.hand_pcd.rotate(self.hand_pcd.get_rotation_matrix_from_xyz((np.pi, np.pi, 0)))
        return self.hand_pcd, self.hand_lineset

    def rotate_landmarks(self, thetas, rotation_axes):
        rotated_landmarks = list()
        for landmark, theta, axis in zip(self.landmarks, thetas, rotation_axes):
            R = Rodrigues_matrix(theta, axis)
            rotated_landmarks.append(np.dot(R, landmark))
        return np.array(rotated_landmarks)


    def rotate_landmarks_v2(self, thetas, rotation_axes):
        rotated_landmarks = self.landmarks.copy()
        # get the vectors needed to rotate
        for (start_connection, end_connection), theta, axes in zip(keypoint_connections, thetas, rotation_axes):
            # get rotation matrix
            R = Rodrigues_matrix(theta, axes)
            # rotate vector and add it
            pivot = end_connection.start
            # rotate the children vectors, following forward kinamatics
            # get all the children of the pivot in a np.array and rotate around the pivot
            indexes = dfs(pivot, CONNECTIONS_DAG)
            rotated_landmarks[indexes, :] = rotated_landmarks[indexes, :] - rotated_landmarks[pivot]
            rotated_landmarks[indexes, :] = rotated_landmarks[indexes, :] @ R
            rotated_landmarks[indexes, :] += rotated_landmarks[pivot]

        return rotated_landmarks


def dfs(node, graph):
    index = list()
    for child in graph[node]:
        index.append(child)
        index.extend(dfs(child, graph))
    
    return index

def hand_lineset(landmarks):
    # created points
    colors = color_bar(len(landmarks))
    num_landmarks = len(landmarks)
    landmarks = o3d.utility.Vector3dVector(landmarks)

    landmark_pcd = o3d.geometry.PointCloud(landmarks)
    landmark_pcd.colors = colors

    # create lines
    lines = list()
    for connection in Connections.HAND_CONNECTIONS:
        lines.append((connection.start, connection.end))
    lines = o3d.utility.Vector2iVector(lines)

    # return checkboard lineset
    landmark_lineset = o3d.geometry.LineSet(points=landmarks, lines=lines)
    landmark_lineset.colors = color_bar(len(lines))
    return [landmark_lineset, landmark_pcd]

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

def color_bar(length):
    # color_variations = color_bar_np(length) * 255
    color_variations = [[0, 255, 0]] * length
    return o3d.utility.Vector3dVector(color_variations)


class StreamingVisualizer(Visualizer, SyncRedisConsumer):
    def __init__(self, cams=2, R=[], T=[]):
        Visualizer.__init__(self, cams, R, T)
        SyncRedisConsumer.__init__(self, "channel_points_3d")
        

    def consume(self) -> bool:
        self.display_scene()
        for message in self.pubsub.listen():
            message = self.convert_messages(message)
            # if a terminate message comes thru return false
            # gracefully close window and unsub
            if message.terminate:
                break

            # update landmarks and keep rendering
            self.update_scene(message.points)
        self.viz.run()
    
    def replay_angles(self, angle_records):
        self.display_scene()
        idx = 0
        sleep_time = (angle_records[-1].timestamp).total_seconds() / len(angle_records)
        for idx in range(len(angle_records)):
            # if idx > len(angle_records):
            #     break
            record = angle_records[idx % len(angle_records)]
            print("updating... {}".format(idx), record.timestamp)
            rotated_landmarks = self.hands.rotate_landmarks_v2(record.thetas, record.rotation_axes)
            self.update_scene(rotated_landmarks)
            time.sleep(sleep_time)

        

class HandStreamingVisualizer(Visualizer, SyncRedisConsumer):
    """
    A class for visualizing hand landmarks in 3D space, streamed from Redis and rendered using a Visualizer.

    Attributes:
        cams (int): The number of cameras used in the visualization.
        R (list): A list of rotation matrices for each camera.
        T (list): A list of translation vectors for each camera.
    """
    def __init__(self, cams=2, R=[], T=[]):
        Visualizer.__init__(self, cams, R, T)
        SyncRedisConsumer.__init__(self, "channel_points_3d")
        

    def consume(self) -> bool:
        """
        Listens for Redis messages containing hand landmarks, updates the visualization, and saves measurements.

        Returns:
            bool: Always returns False once terminated or measurement count exceeds 500.
        """
        count = list()
        self.display_scene()
        for message in self.pubsub.listen():
            message = self.convert_messages(message)
            # if a terminate message comes thru return false
            # gracefully close window and unsub
            if message.terminate:
                break

            # update landmarks and keep rendering
            points = np.array(message.points)
            if points.shape != (21, 3):
                continue
            self.update_scene(points.tolist())
            count.append(measure_hand_connections(points))
            # if len(count) > 500:
            #     np.save("world_hand_measurements", count)
            #     break
            print("updates scene", points.shape)
        self.viz.run()

    def get_world_landmarks(self, message):
        world_landmarks = message.result.hand_world_landmarks
        landmarks = list()
        for wl in world_landmarks:
            for landmark in wl:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return np.array(landmarks)

def measure_hand_connections(pcd: np.ndarray):
    
    middle_start_index = Connections.HAND_MIDDLE_FINGER_CONNECTIONS[0].start
    middle_end_index = Connections.HAND_MIDDLE_FINGER_CONNECTIONS[-1].end
    
    index_start_index = Connections.HAND_INDEX_FINGER_CONNECTIONS[0].start
    index_end_index = Connections.HAND_INDEX_FINGER_CONNECTIONS[-1].end

    ring_start_index = Connections.HAND_RING_FINGER_CONNECTIONS[0].start
    ring_end_index = Connections.HAND_RING_FINGER_CONNECTIONS[-1].end

    pinky_start_index = Connections.HAND_PINKY_FINGER_CONNECTIONS[0].start
    pinky_end_index = Connections.HAND_PINKY_FINGER_CONNECTIONS[-1].end

    thumb_start_index = Connections.HAND_THUMB_CONNECTIONS[0].start
    thumb_end_index = Connections.HAND_THUMB_CONNECTIONS[-1].end
    res = [
        1.5 * np.linalg.norm(pcd[thumb_end_index] - pcd[thumb_start_index]),
        1.5 * np.linalg.norm(pcd[index_start_index] - pcd[index_end_index]),
        1.5 * np.linalg.norm(pcd[middle_end_index] - pcd[middle_start_index]), 
        1.5 * np.linalg.norm(pcd[ring_end_index] - pcd[ring_start_index]),
        1.5 * np.linalg.norm(pcd[pinky_end_index] - pcd[pinky_start_index]),
    ]

    return res

class HandlandmarkVideoStream:
    def __init__(self, cam_id) -> None:
        self.cam_id = cam_id
        
        remaps = np.load("camera_extrinsics/stereo_rectification/stereo_rectification_maps.npz")
        if self.cam_id == 0:
            self.remap_x, self.remap_y = remaps['left_map_x'], remaps['left_map_y']
        elif self.cam_id == 1:
            self.remap_x, self.remap_y = remaps['right_map_x'], remaps['right_map_y']

    def consume(self, frame, message):
        landmark_coords = landmarks_to_numpy(message_objs=[message])
        # landmark_coords = self.rectify_landmarks(landmark_coords)
        landmark_frame = self.draw_landmarks(frame, landmark_coords)
        return landmark_frame
    
    def rectify_landmarks(self, landmarks):
        landmarks = landmarks.reshape((-1, 2))
        # Convert to integer indices for remap lookup
        x_rect = np.round(landmarks[:, 0]).astype(int)
        y_rect = np.round(landmarks[:, 1]).astype(int)

        # Use the remap arrays to get original coordinates
        x_orig = self.remap_x[y_rect, x_rect]
        y_orig = self.remap_y[y_rect, x_rect]

        # Combine into (21, 2) shape
        return np.stack([x_orig, y_orig], axis=1)


    def draw_landmarks(self, frame, landmarks):
        landmarks = np.int32(landmarks)
        colors = color_bar_np(landmarks.shape[0])
        for idx, landmark in enumerate(landmarks):
            cv.circle(frame,landmark, radius=5, color=colors[idx], thickness=1)
        
        for idx, connection in enumerate(Connections.HAND_CONNECTIONS):
            cv.line(frame, landmarks[connection.start], landmarks[connection.end], color=colors[idx], thickness=5)
        
        return frame
