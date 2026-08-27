import minidiff as md
from minidl.layers import ActivationLayer, Dense
from minidl.neural_networks import NeuralNetwork
from minidl.optimizers import AdamW


# Burgers' equation: du/dt + u * du/dx - v * d^2u/dx^2 = 0
def get_train_resources():
    with md.no_grad():
        v = 0.01 / 3.14159265

        # collocation points:
        N_collocation = 5000
        N_initials = 200
        N_boundaries = 200
        N_points = N_collocation + N_initials + N_boundaries
        x_f = md.rand(N_collocation) * 2 - 1
        t_f = md.rand(N_collocation)

        # initial conditions:
        x_ic = md.rand(N_initials) * 2 - 1
        t_ic = md.zeros(N_initials)
        u_ic = -md.sin(3.14159265 * x_ic)

        # boundary conditions:
        x_bc = md.choice(md.Tensor([-1, 1]), size=N_boundaries)
        t_bc = md.rand(N_boundaries)
        u_bc = md.zeros(N_boundaries)

        all_indices = md.permutation(N_points)
        collocation_indices = all_indices[:N_collocation]
        initial_indices = all_indices[N_collocation : N_collocation + N_initials]
        boundary_indices = all_indices[N_collocation + N_initials :]

        ds = md.zeros((N_points, 2), allow_grad=True)
        ds[collocation_indices, 0] = x_f
        ds[collocation_indices, 1] = t_f
        ds[initial_indices, 0] = x_ic
        ds[initial_indices, 1] = t_ic
        ds[boundary_indices, 0] = x_bc
        ds[boundary_indices, 1] = t_bc

    def mse_loss(u: md.Tensor) -> md.Tensor:
        u.backward(allow_higher_order=True, cleanup_mode="destroy")
        dx, dt = ds.grad.T
        dx.backward(allow_higher_order=True)
        dxx = ds.grad[:, 0]

        residual = dt + u.ravel() * dx - v * dxx
        L_r = md.mean(residual**2)
        L_ic = md.mean((u_ic - u[initial_indices].ravel()) ** 2)
        L_bc = md.mean((u_bc - u[boundary_indices].ravel()) ** 2)

        return L_r + L_ic + L_bc

    return ds, mse_loss


def train_PINN(model: NeuralNetwork, optimizer, epochs=5000):
    assert model.trainable
    model.setup_layers(optimizer)

    for epoch in range(epochs):
        data, loss_fn = get_train_resources()
        prediction = model(data)
        loss = loss_fn(prediction)
        loss.backward(cleanup_mode="destroy")

        print(f"Epoch #{epoch}: {loss.item()}")
        with md.no_grad():
            model.update_layer_weights()


def main():
    model = NeuralNetwork()
    model.trainable = True
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
    train_PINN(model, AdamW(learning_rate=0.001))
    model.save_network("out/burgers_pinn.npy")


if __name__ == "__main__":
    main()
