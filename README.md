# PINN for Solving a First-Order ODE

This project serves as a foundational example of a Physics-Informed Neural Network (PINN) implemented in PyTorch. It demonstrates how to train a neural network to solve a simple first-order Ordinary Differential Equation (ODE), `du/dt = cos(2πt)`, with a given initial condition, `u(0) = 1`. The goal is to approximate the unknown function `u(t)` by encoding the differential equation directly into the network's loss function.

---

## 📜 Project Description

Physics-Informed Neural Networks are a class of neural networks that learn to solve differential equations by satisfying the physical laws they represent. Instead of relying solely on data, a PINN's loss function includes terms that measure how well the network's output satisfies the governing equations and boundary conditions. This project provides a clear, step-by-step implementation for a basic ODE, making it an excellent starting point for understanding the core principles of PINNs.

---

## ✨ Key Features

-   **Neural Network Approximator**: A simple feedforward neural network serves as a universal function approximator for the solution `u(t)`.
-   **Physics-Informed Loss**: The custom loss function combines two components:
    1.  The residual from the ODE (`du/dt - cos(2πt)`).
    2.  The error from the initial condition (`u(0) - 1`).
-   **Automatic Differentiation**: Leverages PyTorch's `autograd` feature to compute the derivative `du/dt` required for the ODE residual, eliminating the need for manual numerical differentiation.
-   **Validation**: Compares the PINN's prediction against the exact analytical solution to visually assess its accuracy.

---

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

-   Python 3.8+
-   PyTorch
-   NumPy
-   Matplotlib

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required packages**:
    ```bash
    pip install torch numpy matplotlib
    ```

---

## 🛠️ Usage

1.  **Run the script**:
    Execute the Python script from your terminal.
    ```bash
    python main.py
    ```

2.  **Review the output**:
    -   The training loss will be printed to the console at regular intervals.
    -   A plot will be generated showing the PINN's predicted solution alongside the exact analytical solution for comparison.

---

## 💻 Code Structure

The entire implementation is contained within a single Python script (`main.py`) for simplicity and clarity. The key steps are:

1.  **PINN Class Definition**: A `torch.nn.Module` defines the neural network architecture.
2.  **Data Generation**: `torch.linspace` is used to create collocation points where the ODE will be enforced.
3.  **Loss Function (`pinn_loss`)**: A function that calculates the combined ODE and initial condition loss.
4.  **Training Loop**: A standard training loop that uses an optimizer (like SGD or Adam) to minimize the loss function and update the network's weights.
5.  **Results Visualization**: After training, the model's output is plotted against the true solution using Matplotlib.

---

## 🤝 Contributing

Feedback and contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

---

## 📄 License

This project is open-source and available under the MIT License.
