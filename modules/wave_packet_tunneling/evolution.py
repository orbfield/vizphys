import numpy as np

class WaveFunctionEvolution:
    @staticmethod
    def evolve_wavefunction(psi, x, dt, V_total):
        """Evolve the wave function using the split-operator method."""
        dx = x[1] - x[0]
        N = len(x)
        
        # Momentum space grid
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        
        # Kinetic energy operator in momentum space
        T = (k**2) / 2
        
        # Precompute exponentials
        exp_T = np.exp(-1j * T * dt / 2)
        exp_V = np.exp(-1j * V_total * dt)
        
        # Split-operator method steps
        psi_k = np.fft.fft(psi)
        psi_k *= exp_T
        psi = np.fft.ifft(psi_k)
        psi *= exp_V
        psi_k = np.fft.fft(psi)
        psi_k *= exp_T
        psi = np.fft.ifft(psi_k)
        
        return psi