import numpy as np

class WaveFunction:
    @staticmethod
    def calculate_kappa_n(n):
        """Calculate κₙ using m³ structure where n = m²."""
        m = int(np.sqrt(n))
        if m**2 != n:
            raise ValueError(f"n must be a perfect square. Received n={n}")
        return np.pi * (m ** 3)

    @staticmethod
    def wavefunction_1d(x, n, x0=0.0):
        """
        Calculate the 1D wavefunction as a superposition of two plane waves.
        """
        kappa = WaveFunction.calculate_kappa_n(n)
        alpha = 0.0126  # Gaussian width parameter
        shifted_x = x - x0
        wave = np.exp(1j * kappa * shifted_x)
        gaussian = np.exp(-alpha * shifted_x**2 / 2)
        psi = gaussian * wave
        
        # Normalize the wavefunction
        dx = x[1] - x[0]
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        return psi / norm

    @staticmethod
    def initialize_wavefunction(x, n, x0=-100.0, sigma=0.8):
        """Initialize a custom Gaussian-modulated wave packet."""
        return WaveFunction.wavefunction_1d(x, n, x0)

    @staticmethod
    def compute_total_probability(psi, dx):
        """Compute the total probability density."""
        return np.sum(np.abs(psi)**2) * dx