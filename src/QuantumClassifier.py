# variational quantum classifier --> learning binary classification patterns in two-dimensional data spaces
import pennylane as qml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data.datasets import Dataset
torch.set_default_dtype(torch.float64)

dataset = Dataset("nonlinear1")

quantum_classifier = qml.device('default.qubit', wires=2) # wires [0, 1] (2 qubits)

@qml.qnode(quantum_classifier, interface = 'torch')
def circuit(quantum_params, feature_vector):
    # data_encoding: (encoding classical data, feature_vector, into quantum states through a parameterized circuit that will run on the quantum_classifier device)
    qml.AngleEmbedding(features=feature_vector, wires=range(2), rotation='Y') # encoding features on [0, 1] qubits using Y-rotations
    qml.Hadamard(0)

    #  Variational ansatz:
    qml.StronglyEntanglingLayers(weights=quantum_params, wires=range(2)) # Entanglement pattern in the variational circuit, where the weights (params) are the adjustable parameters of the circuit.

    # Measurement:
    return (qml.expval(qml.PauliZ(0)), 
            qml.expval(qml.PauliZ(1))) # expectation values of the Pauli-Z operator on both qubits, which will be used for classification.

shape = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=2)
rng = np.random.default_rng(12345)
weights = rng.random(size=shape)

# Rule that combines the obserble values of the circuit for classification:
def model(quantum_params, classical_params, x): 
    z0, z1 = circuit(quantum_params, x) # output of the quantum circuit
    
    w = classical_params[:2] # The classical params are the one used in the linear combination of the circuit outputs
    b = classical_params[2]
    
    # Linear combination:
    s = (w[0] * z0) + (w[1] * z1) + b
    return torch.sigmoid(s) # condenses the number between 0 and 1

def cost(quantum_params, classical_params, X, Y): # X = classical data
    loss = 0                                      # Y = labelled data in dataset
    correct = 0
    for xi, yi in zip(X, Y):
        ModelPrediction = model(quantum_params, classical_params, xi) # between 0 and 1, probability of belonging to class 1
        loss += F.binary_cross_entropy(ModelPrediction, yi)

        # Class decision itself: (classification based on the output of the model)
        if ModelPrediction.item() >= 0.5:
            Class = 1
        else:
            Class = 0
        if Class == int(yi.item()):
            correct += 1

    return (loss / len(X), correct / len(X)) # average loss and accuracy over the dataset

def PlotFunctions(loss_history, accuracy_history):
    fig, ax1 = plt.subplots()

    # Loss (left axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.plot(loss_history, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="red")
    ax2.plot(accuracy_history, color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    if dataset.getName() == "linear":
        plt.title("VQC Training: Loss and Accuracy - Make Classification dataset (linear)")
    elif dataset.getName() == "nonlinear0":
        plt.title("VQC Training: Loss and Accuracy - Make Moons dataset (nonlinear)")
    elif dataset.getName() == "nonlinear1":
        plt.title("VQC Training: Loss and Accuracy - Make Circles dataset (nonlinear)")
    plt.show()

# Randomizing the initial parameters:
quantum_params = torch.tensor(weights, requires_grad=True)
classical_params = torch.randn(3, requires_grad=True)

# Optimizer:
optimizer = torch.optim.Adam([quantum_params, classical_params], lr=0.05)
epochs = 100
loss_history = []
accuracy_history = []

# Training loop:
X = torch.tensor(dataset.X(), dtype=torch.float64)
Y = torch.tensor(dataset.Y(), dtype=torch.float64)
for epoch in range(epochs):
    optimizer.zero_grad() # reset the gradients
    loss, accuracy = cost(quantum_params, classical_params, X, Y)
    loss.backward() # parameter-shift rule and computational graph to calculate the gradients and perform backpropagation
    optimizer.step() # updating both quantum and classical parameters based on the computed gradients to minimize the cost function

    loss_history.append(loss.item())
    accuracy_history.append(accuracy)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

# The model is trained, we can now visualize the convergence of the cost function and accuracy over epochs:
PlotFunctions(loss_history, accuracy_history)