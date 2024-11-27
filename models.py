import numpy as np
from scipy.linalg import expm, logm

SigX = np.array([[0, 1], [1, 0]])
SigY = np.array([[0, -1j], [1j, 0]])
SigZ = np.array([[1, 0], [0, -1]])
SigI = np.array([[1, 0], [0, 1]])

pauli_basis_2q = [
    np.kron(SigI, SigI),
    np.kron(SigI, SigX),
    np.kron(SigI, SigY),
    np.kron(SigI, SigZ),
    np.kron(SigX, SigI),
    np.kron(SigX, SigX),
    np.kron(SigX, SigY),
    np.kron(SigX, SigZ),
    np.kron(SigY, SigI),
    np.kron(SigY, SigX),
    np.kron(SigY, SigY),
    np.kron(SigY, SigZ),
    np.kron(SigZ, SigI),
    np.kron(SigZ, SigX),
    np.kron(SigZ, SigY),
    np.kron(SigZ, SigZ),
]

len(pauli_basis_2q)

ux1 = np.kron(expm(-1j * np.pi / 4 * SigX), SigI)
ux2 = np.kron(SigI, expm(-1j * np.pi / 4 * SigX))
uz1 = np.kron(expm(-1j * np.pi / 4 * SigZ), SigI)
uz2 = np.kron(SigI, expm(-1j * np.pi / 4 * SigZ))
uy1 = np.kron(expm(-1j * np.pi / 4 * SigY), SigI)
uy2 = np.kron(SigI, expm(-1j * np.pi / 4 * SigY))


def model_CZ(x):
    generator = np.zeros((4, 4), dtype=complex)
    generator += (np.pi/2)*(pauli_basis_2q[3] + pauli_basis_2q[12]) - np.pi/2*(pauli_basis_2q[15] + pauli_basis_2q[0])
    for i in range(15):
        generator += x[i]*pauli_basis_2q[i+1]
    return expm(-(1j/2)*generator)

class UnitaryGateSetModel_CZOnly:
    def __init__(self):
        self.num_qubits = 2
        self.num_params = 15
        self.gate_names = ['CZ', 'X1', 'X2', 'Z1', 'Z2', 'Y1', 'Y2']
        self.hilbert_dims = 4
        self.cartan_basis = [
            np.diag([1, -1, 0, 0]), 
            np.diag([0, 1, -1, 0]),
            np.diag([0, 0, 1, -1]),
        ]

    def CZ(self, parameters):
        return model_CZ(parameters)
    
    def X1(self, parameters):
        return np.kron(expm(-1j * np.pi / 4 * SigX), SigI)
    
    def X2(self, parameters):
        return np.kron(SigI, expm(-1j * np.pi / 4 * SigX))
    
    def Z1(self, parameters):
        return np.kron(expm(-1j * np.pi / 4 * SigZ), SigI)
    
    def Z2(self, parameters):
        return np.kron(SigI, expm(-1j * np.pi / 4 * SigZ))
    
    def Y1(self, parameters):
        X1 = self.X1(parameters)
        Z1 = self.Z1(parameters)
        return Z1 @ Z1 @ Z1 @ X1 @ Z1 
    
    def Y2(self, parameters):
        X2 = self.X2(parameters)
        Z2 = self.Z2(parameters)
        return Z2 @ Z2 @ Z2 @ X2 @ Z2
    
    def gate(self, name, parameters):
        if name not in self.gate_names:
            raise ValueError(f"Gate {name} not found in the model")
        return getattr(self, name)(parameters)
    
    def target_gate(self, name):
        if name not in self.gate_names:
            raise ValueError(f"Gate {name} not found in the model")
        return getattr(self, name)(np.zeros(self.num_params))
    
    def compile_unitary(self, germ, parameters):
        u = np.eye(self.hilbert_dims)
        for gate_name in germ:
            u = self.gate(gate_name, parameters) @ u
        return u  
    
    def lie_basis(self, idx):
        return pauli_basis_2q[idx+1]
    
    # def true_cartan_parameters(self, germ, params):
    #     unitary = self.compile_unitary(germ, params)
    #     log_unitary = logm(unitary)
    #     cparams = np.array( [np.trace(log_unitary @ basis) for basis in self.cartan_basis] )
    #     # check that parameters are imaginary
    #     assert np.allclose(np.real(cparams), 0)
    #     return np.imag(cparams)