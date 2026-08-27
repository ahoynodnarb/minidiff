import mlx.core as mx
import numpy as np
from minidl.layers import ActivationLayer, Dense
from minidl.neural_networks import NeuralNetwork

import minidiff as md


def main():
    model = NeuralNetwork()
    model.set_layers(
        Dense(32, 2),
        ActivationLayer(md.tanh),
        Dense(32, 32),
        ActivationLayer(md.tanh),
        Dense(32, 32),
        ActivationLayer(md.tanh),
        Dense(32, 32),
        ActivationLayer(md.tanh),
        Dense(1, 32),
    )
    model.load_network("out/burgers_pinn_keep.npy")
    with np.load("data/Burgers.npz") as data:
        x_np, t_np = data["x"], data["t"]
        u = md.Tensor(data["usol"].T.ravel())
        combined = np.vstack(np.meshgrid(x_np, t_np)).reshape(2, -1).T
        combined = md.Tensor(combined)

    out = model(combined).ravel()
    print(md.mean((out - u) ** 2))


if __name__ == "__main__":
    md.backend.set_backend(md.backend.numpy)
    main()
