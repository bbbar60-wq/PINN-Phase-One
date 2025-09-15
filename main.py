# Code Cell 1: Imports
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# For reproducibility
torch.manual_seed(0)

print("Libraries imported: torch, nn, optim, matplotlib, numpy.")

# Code Cell 2: NN Model Definition
class PINN(nn.Module):
    def __init__(self, neurons=20):
        super(PINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 1)
        )

    def forward(self, t):
        return self.net(t)

# Create an instance of the model
model = PINN(neurons=20)

print("PINN model created with 2 hidden layers, 20 neurons each.")

# Code Cell 3: Generate Collocation Points
N = 50
t_colloc = torch.linspace(0.0, 1.0, N).unsqueeze(1)  # shape: (N,1)
t_colloc.requires_grad = True  # We need gradients wrt t

print("Collocation points t_colloc:")
print(t_colloc[:5], "...")

# Code Cell 4: Defining the PINN Loss Function
def pinn_loss(model, t_vals):
    # Forward pass: get u(t)
    u_hat = model(t_vals)  # shape (N,1)

    # Compute du/dt via autograd
    # grad_outputs must have the same shape as u_hat
    dudt = torch.autograd.grad(
        u_hat, t_vals,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True
    )[0]

    # ODE residual: du/dt - cos(2 pi t)
    # We'll compute MSE of that residual
    ode_res = dudt - torch.cos(2.0 * torch.pi * t_vals)
    ode_loss = torch.mean(ode_res**2)

    # Initial condition: u(0) = 1
    # We can just evaluate model at t=0
    # Alternatively, we might gather the first collocation point if it's 0
    u0_hat = model(torch.tensor([[0.0]], requires_grad=True))
    bc_loss = (u0_hat - 1.0)**2  # scalar

    # Total loss
    return ode_loss + bc_loss

# Code Cell 5: Training the PINN using Gradient Descent (GD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3000

for epoch in range(num_epochs):
    # Compute the total loss (ODE + IC)
    loss_value = pinn_loss(model, t_colloc)

    # Zero gradients, backprop, and step
    optimizer.zero_grad()  # Reset gradients
    loss_value.backward()  # Compute gradients (backpropagation)
    optimizer.step()  # Update parameters using gradient descent

    # Print progress every 500 epochs
    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss = {loss_value.item():.6e}")

# Code Cell 6: Plot and Compare
t_test = torch.linspace(0.0, 1.0, 200).unsqueeze(1)
u_pred = model(t_test).detach().numpy()

# Exact solution for reference
t_test_np = t_test.detach().numpy()
u_exact = (1.0/(2*np.pi))*np.sin(2*np.pi*t_test_np) + 1.0

# Plot
plt.figure(figsize=(7,4))
plt.plot(t_test_np, u_exact, 'b', label='Exact solution')
plt.plot(t_test_np, u_pred, 'r--', label='PINN Prediction')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN for du/dt = cos(2πt),  u(0)=1')
plt.legend()
plt.grid(True)
plt.show()