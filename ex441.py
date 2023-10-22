import numpy as np

def phase_normalized_output(v):
    if not np.allclose(v[0], 0):
        v = v * np.conj(v[0])
    else:   
        v = v * np.conj(v[1])
    return v / np.linalg.norm(v)

def measurement_probabilities(circuit, input):
    res = np.absolute(circuit @ input)
    res /= np.linalg.norm(res)
    return res ** 2

H = np.array([1,1,1,-1]).reshape(2, 2) / np.sqrt(2)
S = np.array([1, 0, 0, 1j]).reshape(2, 2)
Z = np.array([1, 0, 0, -1]).reshape(2, 2)
I = np.array([1, 0, 0, 1]).reshape(2, 2)

theta = np.arccos(3 / 5)
Rz = np.array([np.exp(-1j * theta / 2), 0, 0, np.exp(1j * theta / 2)]).reshape(2, 2)

layer_one = np.kron(H, np.kron(H, I))
layer_two = np.eye(8)
layer_two[6:, 6:] = np.array([0, 1, 1, 0]).reshape(2, 2)
layer_three = np.kron(I, np.kron(I, S))

# Circuit for exercise 4.41
circuit = layer_one @ layer_two @ layer_three @ layer_two @ layer_one

c0 = np.array([1, 0]).reshape(2, 1) # |0>
c1 = np.array([0, 1]).reshape(2, 1) # |1>
k000 = np.kron(c0, np.kron(c0, c0)) # |000>
k001 = np.kron(c0, np.kron(c0, c1)) # |001>

print("---")

# Probabilities for computational basis:
print(measurement_probabilities(circuit, k000))
print(measurement_probabilities(circuit, k001))

print("---")

# Phase normalized outputs for |000>:
res = circuit @ k000
print("RAW", res)
print("(|00.>):", phase_normalized_output(res[:2]))
print("(|01.>):", phase_normalized_output(res[2:4]))
print("(|10.>):", phase_normalized_output(res[4:6]))
print("(|11.>):", phase_normalized_output(res[6:]))
print("Pure Rotation:", phase_normalized_output(Rz @ c0))
print("Pure Z:", phase_normalized_output(Z @ c0))

print("---")

# Phase normalized outputs for |001>:
res = circuit @ k001
print("RAW", res)
print("(|00.>):", phase_normalized_output(res[:2]))
print("(|01.>):", phase_normalized_output(res[2:4]))
print("(|10.>):", phase_normalized_output(res[4:6]))
print("(|11.>):", phase_normalized_output(res[6:]))
print("Pure Rotation:", phase_normalized_output(Rz @ c1))
print("Pure Z:", phase_normalized_output(Z @ c1))
