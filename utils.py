import logging
import gc
import os

import h5py
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from .trajectory import load_weights, weight_encoder


def get_vectors(model, seed=None, trajectory=None):
    """
    Updated get_vectors function to handle new TF/Keras requirements.
    """
    np.random.seed(seed)
    vector_x, vector_y = list(), list()
    weights = model.get_weights()

    if trajectory:
        # re-load weights from trajectory
        load_weights(model, trajectory)
        file_path = os.path.join(trajectory, ".trajectory", "model_weights.weights.h5")  # Renamed here

        with h5py.File(file_path, "r+") as f:
            differences = list()
            trajectory_array = np.array(f["weights"])
            for i in range(0, len(trajectory_array) - 1):
                differences.append(trajectory_array[i] - trajectory_array[-1])

            pca = PCA(n_components=2)
            pca.fit(np.array(differences))
            f["X"], f["Y"] = pca.transform(np.array(differences)).T

        vector_x = weight_encoder(model, pca.components_[0])
        vector_y = weight_encoder(model, pca.components_[1])

        return weights, vector_x, vector_y

    else:
        # Generate random vectors compatible with the layer sizes
        # Norm-based approach
        cast = np.array([1]).T
        for layer in weights:
            k = len(layer.shape) - 1
            d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)
            dist_x = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)
            vector_x.append(
                (
                    dist_x * (cast * np.linalg.norm(layer, axis=k))[:, np.newaxis]
                ).reshape(d.shape)
            )

            d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)
            dist_y = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)
            vector_y.append(
                (
                    dist_y * (cast * np.linalg.norm(layer, axis=k))[:, np.newaxis]
                ).reshape(d.shape)
            )

        return weights, vector_x, vector_y


def _obj_fn(model, data, solution):
    """
    Simple wrapper to evaluate a model's performance on given data
    with temporarily modified weights.
    """
    old_weights = model.get_weights()
    model.set_weights(solution)
    value = model.evaluate(data[0], data[1], verbose=0)
    model.set_weights(old_weights)
    return value


def build_mesh(
    model,
    data,
    grid_length,
    extension=1,
    filename="meshfile",
    verbose=True,
    seed=None,
    trajectory=None,
):
    """
    Builds a mesh of loss/metrics values varying along two principal directions
    in weight space. Saves the results in an HDF5 file.
    """
    logging.basicConfig(level=logging.INFO)

    # model.metrics_names generally works in tf.keras, but make sure
    # your model is compiled and has metrics defined.
    z_keys = model.metrics_names
    z_keys[0] = model.loss if hasattr(model, 'loss') else 'loss'
    Z = list()

    # get vectors and set spacing
    origin, vector_x, vector_y = get_vectors(model, seed=seed, trajectory=trajectory)
    space = np.linspace(-extension, extension, grid_length)
    X, Y = np.meshgrid(space, space)

    for i in range(grid_length):
        if verbose:
            logging.info("line {} out of {}".format(i, grid_length))

        for j in range(grid_length):
            solution = [
                origin[x] + X[i][j] * vector_x[x] + Y[i][j] * vector_y[x]
                for x in range(len(origin))
            ]
            Z.append(_obj_fn(model, data, solution))

    Z = np.array(Z)
    os.makedirs("./files", exist_ok=True)

    # Save results
    with h5py.File(f"./files/{filename}.hdf5", "w") as f:
        f["space"] = space
        original_results = _obj_fn(model, data, origin)

        for i, metric in enumerate(z_keys):
            f[f"original_{metric}"] = original_results[i]
            f[metric] = Z[:, i].reshape(X.shape)

    del Z
    gc.collect()