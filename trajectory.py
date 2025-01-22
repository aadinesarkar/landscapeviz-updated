import h5py
import numpy as np
import os
import tensorflow as tf


def load_weights(model, folder_path):
    """
    Loads all .hdf5 files in folder_path into a single HDF5 file
    called model_weights.weights.h5. Then compiles them into
    an array named 'weights' for reference.
    """

    sgd_weights = list()
    traj_dir = os.path.join(folder_path, ".trajectory")
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)
    file_path = os.path.join(traj_dir, "model_weights.weights.h5")  # Renamed to .weights.h5

    # Gather all .hdf5 weight files sorted by name
    for weights_file in sorted(os.listdir(folder_path)):
        if weights_file.endswith(".hdf5"):
            model.load_weights(os.path.join(folder_path, weights_file))
            solution = weight_decoder(model)
            sgd_weights.append(solution)

    with h5py.File(file_path, "w") as f:
        f["weights"] = np.array(sgd_weights)


def weight_decoder(model):
    """
    Flattens and concatenates all layer weights into a single 1D numpy array.
    """
    solution = np.array([])
    weights = model.get_weights()
    for layer_weights in weights:
        solution = np.append(solution, layer_weights.flatten())
    return solution


def weight_encoder(model, solution):
    """
    Reshapes a 1D array into the shapes of the model's layers
    and assigns them as new weights.
    """
    start = 0
    weights = model.get_weights()
    for i in range(len(weights)):
        weight_shape = weights[i].shape
        finish = np.prod(weight_shape)
        weights[i] = np.reshape(solution[start : start + finish], weight_shape)
        start += finish
    return weights