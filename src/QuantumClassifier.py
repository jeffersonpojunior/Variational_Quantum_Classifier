# classificador quântico variacional --> aprender padrões de classificação binária em espaços de dados bidimensionais
import pennylane as qml
from jax import numpy as np
import jax
import jaxopt
import matplotlib.pyplot as plt

quantum_classifier = qml.device("lightning.qubit", wires = 2)

@qml.qnode(quantum_classifier)
def variational_classifier(params, x):
    # data_encoding: (codificação de dados clássicos em estados quânticos através de um circuito parametrizado U(x))
    qml.RX(x[0], wires = 0)
    qml.RX(x[1], wires = 1)

    # Ansatz Variacional:
    qml.CNOT(wires = [0, 1])
    qml.RY(params[0], wires = 0)
    qml.RY(params[1], wires = 1)

    # Medição:
    return qml.expval(qml.Z(0))

gradient = jax.grad(variational_classifier, argnums = 0)
print(gradient(np.array([0.5, 0.5]), np.array([1.0, 1.0])))

# Otimização de Parâmetros:
init_params = np.array([0.0, 0.0])
def cost(params):
    return variational_classifier(params, np.array([1.0, 1.0]))

opt = jaxopt.GradientDescent(cost, stepsize = 0.4, acceleration=False)
steps = 100
params = init_params

opt_state = opt.init_state(params)

for i in range(steps):
    params, opt_state = opt.update(params, opt_state)

print(f"Optimized parameters: {params}")