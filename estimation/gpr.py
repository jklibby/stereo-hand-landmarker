import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from parallelism import RedisStreamConsumer
from model.landmarker import KeyPointAngles

def load_streams(streams=[]):
    rsc = RedisStreamConsumer()

    stream_keypoints = list()
    stream_thetas = list()
    stream_axes = list()
    min_len = float('inf')
    for stream in streams:
        keypoints = rsc.consume(stream)
        min_len = min(min_len, len(keypoints))
        stream_keypoints.append(keypoints)
    
    for keypoints in stream_keypoints:
        axes = list()
        angles = list()
        for keypoint in keypoints[:min_len]:
            axes.append(keypoint.rotation_axes)
            angles.append(keypoint.thetas)
        stream_thetas.append(angles)
        stream_axes.append(axes)
    return np.array(stream_thetas), np.array(stream_axes)

def fit_and_combine_sequences(sequences, n_points=100):
    """
    Fit Gaussian Process Regressors to N sequences with multiple dimensions.
    
    Parameters:
    -----------
    sequences : list of array-like or 3D numpy array
        Input sequences of shape (n_sequences, seq_length, n_dims)
    n_points : int
        Number of points for prediction (default: 100)
        
    Returns:
    --------
    tuple
        Combined sequence, list of GPR predictions, and x values
    """
    if isinstance(sequences, list):
        sequences = np.array(sequences)
    
    n_sequences = sequences.shape[0]
    seq_length = sequences.shape[1]
    n_dims = sequences.shape[2] if len(sequences.shape) > 2 else 1
    
    if len(sequences.shape) == 2:
        sequences = sequences.reshape(n_sequences, seq_length, 1)
    
    X = np.linspace(0, 1, seq_length).reshape(-1, 1)
    X_pred = np.linspace(0, 1, n_points).reshape(-1, 1)
    
    all_predictions = np.zeros((n_sequences, n_points, n_dims))
    all_sigmas = np.zeros((n_sequences, n_points, n_dims))
    
    for i in range(n_sequences):
        for d in range(n_dims):
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
            
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=i*n_dims + d)
            gpr.fit(X, sequences[i, :, d])
            
            y_pred, sigma = gpr.predict(X_pred, return_std=True)
            all_predictions[i, :, d] = y_pred
            all_sigmas[i, :, d] = sigma
    
    weights = 1 / (all_sigmas ** 2)
    weighted_sum = np.sum(weights * all_predictions, axis=0)
    weight_sum = np.sum(weights, axis=0)
    combined_seq = weighted_sum / weight_sum
    
    return combined_seq, all_predictions, X_pred.ravel()

def predict_gpr(thetas, axes):
    stream_count = thetas.shape[0]
    sequence_length = thetas.shape[1]
    num_sequences = thetas.shape[2]

    predicted_thetas = list()
    predicted_axes = list()
    for seq_idx in range(num_sequences):
        combined_thetas, _, _ = fit_and_combine_sequences(thetas[:, :, seq_idx], sequence_length)
        combined_axes, _, _ = fit_and_combine_sequences(axes[:, :, seq_idx, :], sequence_length)
        predicted_thetas.append(combined_thetas)
        predicted_axes.append(combined_axes)
    
    return np.array(predicted_thetas).reshape(sequence_length, 15), np.array(predicted_axes).reshape(sequence_length, 15, 3)

def predicted_gpr_keypoints(thetas, axes):
    length = len(thetas)
    predicted_keypoints = [KeyPointAngles(timestamp=0, thetas=thetas[i, :].tolist(), rotation_axes=axes[i].tolist()) for i in range(length)]
    predicted_keypoints[-1].timestamp = timedelta(seconds=30)
    return predicted_keypoints


def fit_gpr(streams):
    thetas, axes = load_streams(streams)
    combined_thetas, combined_axes = predict_gpr(thetas, axes)
    return predicted_gpr_keypoints(combined_thetas, combined_axes)


