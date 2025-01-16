import numpy as np
from visualization import HandStreamingVisualizer


def _create_stereo_hsv():
    stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")
    viz = HandStreamingVisualizer(cams=2, R=[stereo_calibration["R"]], T=[stereo_calibration["T"]])
    viz.consume()

_create_stereo_hsv()