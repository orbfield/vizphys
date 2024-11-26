import numpy as np

class PotentialBarrier:
    @staticmethod
    def smooth_step(x, edge, width):
        """Create a smooth step function."""
        z = (x - edge) / width
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def create_potential_barrier(x, barrier_center=0.0, V0=20.0, barrier_width=1.0, transition_width=0.05):
        """Create a potential barrier with smooth transitions."""
        V = np.zeros_like(x)
        barrier_start = barrier_center - barrier_width / 2
        barrier_end = barrier_center + barrier_width / 2
        
        V += V0 * (PotentialBarrier.smooth_step(x, barrier_start, transition_width) - 
                   PotentialBarrier.smooth_step(x, barrier_end, transition_width))
        return V

    @staticmethod
    def create_total_potential(x, barrier_center=0.0, V0=20.0, barrier_width=1.0, transition_width=0.05):
        """Create the total potential including the stationary barrier."""
        return PotentialBarrier.create_potential_barrier(x, barrier_center, V0, barrier_width, transition_width)
