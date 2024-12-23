import numpy as np

from visualization import StreamingVisualizer as Visualizer
from model.landmarker import KeyPointAngleRecorder, KeyPointAngles, keypoint_connections
from parallelism.redis import RedisStreamConsumer, RedisDecoder
from datetime import timedelta

def load_extrinsics():
    dir = "calibration"
    stereo_calibration = np.load("camera_extrinsics/stereo_calibration.npz")
    return [stereo_calibration["R"]], [stereo_calibration["T"]]


r, t = load_extrinsics()

# print(r, t)

sv = Visualizer(cams=2, R=r, T=t)

# sv.consume()

decoder = RedisDecoder()
rsc = RedisStreamConsumer()

predicted_angles = np.load('predicted_sequences-stream-1.npy')
axes = np.load('stream-0.npz')['axes']
angles = np.load('stream-0.npz')['angles']
predicted_keypoints = [KeyPointAngles(timestamp=0, thetas=angles[i, :].tolist(), rotation_axes=axes[i].tolist()) for i in range(angles.shape[0])]
predicted_keypoints[-1].timestamp = timedelta(seconds=30)
# for idx, stream in enumerate(['2024-12-08-1733645000', '2024-12-08-1733644569', '2024-12-08-1733642773', '2024-12-08-1733643118']):
#     res = rsc.consume(stream)
#     angles = np.array([r.thetas for r in res])
#     axes = np.array([r.rotation_axes for r in res])
#     np.savez_compressed('stream-{}'.format(idx), angles=angles, axes=axes)


sv.consume_stream(predicted_keypoints)

# '2024-12-08-1733645000' - 5
# 2024-12-08-1733644569 - 4
# '2024-12-08-1733642773' - 4
# 2024-12-08-1733643118 - 4