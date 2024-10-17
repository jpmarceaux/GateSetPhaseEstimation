import numpy as np


class RPEDesign:
    def __init__(self, model, germs, depths):
        self.model = model
        self.germs = germs
        self.depths = depths
        self.num_params = model.num_params
        self.num_qubits = model.num_qubits
        self.hilbert_dims = model.hilbert_dims
        self.cartan_basis = model.cartan_basis
        self.gate_names = model.gate_names

        self.basis_transforms = {
            tuple(germ): self._calculate_basis_transform_at_target(germ) for germ in germs
        }

    def _calculate_basis_transform_at_target(self, germ):
        target_unitary = self.model.compile_unitary(germ, np.zeros(self.num_params))
        _, evecs = np.linalg.eig(target_unitary)
        return evecs 
    
    def inphase_prep(self, germ, subspace):
        a, b = subspace
        plus_ab = np.zeros(self.hilbert_dims, dtype=complex)
        plus_ab[a] = 1/np.sqrt(2)
        plus_ab[b] = 1/np.sqrt(2)
        return self.basis_transforms[tuple(germ)]@plus_ab  
    
    def quadrature_prep(self, germ, subspace):
        a, b = subspace
        plus_ab = np.zeros(self.hilbert_dims, dtype=complex)
        plus_ab[a] = 1/np.sqrt(2)
        plus_ab[b] = 1j/np.sqrt(2)
        return self.basis_transforms[tuple(germ)]@plus_ab
    
    def inphase_prob(self, germ, depth, params, subspace):
        u = self.model.compile_unitary(germ, params)
        u_d = np.linalg.matrix_power(u, depth)
        plus_ab = self.inphase_prep(germ, subspace)
        return np.abs(plus_ab.conj().T @ u_d @ plus_ab)**2
    
    def inphase_prob_series(self, germ, params, subspace):
        return np.array([self.inphase_prob(germ, depth, params, subspace) for depth in self.depths])
    
    def quadrature_prob(self, germ, depth, params, subspace):
        u = self.model.compile_unitary(germ, params)
        u_d = np.linalg.matrix_power(u, depth)
        quad_ab = self.quadrature_prep(germ, subspace)
        plus_ab = self.inphase_prep(germ, subspace)
        return np.abs(plus_ab.conj().T @ u_d @ quad_ab)**2
    
    def quadrature_prob_series(self, germ, params, subspace):
        return np.array([self.quadrature_prob(germ, depth, params, subspace) for depth in self.depths])
    
    def signal(self, germ, params, subspace):
        inphase_probs = self.inphase_prob_series(germ, params, subspace)
        quadrature_probs = self.quadrature_prob_series(germ, params, subspace)
        return (1 - 2*inphase_probs) + 1j*(1 - 2*quadrature_probs)
    
    def target_signal(self, germ, subspace):
        return self.signal(germ, np.zeros(self.num_params), subspace)
    
    