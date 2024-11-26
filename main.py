# main.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules.wave_packet_tunneling.wavefunction import WaveFunction  # Changed from relative import
from modules.wave_packet_tunneling.potential import PotentialBarrier  # Changed from relative import
from modules.wave_packet_tunneling.evolution import WaveFunctionEvolution  # Changed from relative import
from modules.wave_packet_tunneling.visualization import Visualizer  # Changed from relative import

def create_tunneling_animation(
    output_filename='quantum_tunneling.gif',
    num_frames=120,
    spatial_points=2000,
    spatial_range=(-100, 100),
    total_time=6.0,
    barrier_width=1.0,
    V0=20.0,
    transition_width=0.05,
    vis_settings={'width': 2000, 'height': 1000},
    n=1,
    x0=-100.0,
    barrier_center_init=0.0
):
    """Create an animation of quantum tunneling."""
    # Initialize spatial grid
    x = np.linspace(spatial_range[0], spatial_range[1], spatial_points)
    dx = x[1] - x[0]
    
    # Initialize components
    psi = WaveFunction.initialize_wavefunction(x, n, x0)
    V_total = PotentialBarrier.create_total_potential(x, barrier_center_init, V0, barrier_width, transition_width)
    visualizer = Visualizer(vis_settings)
    
    # Calculate time step
    max_k = np.max(np.abs(2 * np.pi * np.fft.fftfreq(spatial_points, d=dx)))
    dt = 0.05 / (max_k**2 / 2)
    
    # Setup time evolution
    num_time_steps = int(total_time / dt) + 1
    times = np.linspace(0, total_time, num_time_steps)
    frame_indices = np.linspace(0, num_time_steps - 1, num_frames).astype(int)
    
    frames = []
    probabilities = []
    
    print("Generating frames...")
    for i in tqdm(range(num_time_steps)):
        # Evolve wave function
        psi = WaveFunctionEvolution.evolve_wavefunction(psi, x, dt, V_total)
        
        # Record probability
        total_prob = WaveFunction.compute_total_probability(psi, dx)
        probabilities.append(total_prob)
        
        # Capture frame if needed
        if i in frame_indices:
            img = visualizer.create_datashader_frame(
                psi, x, 
                barrier_center=barrier_center_init, 
                barrier_width=barrier_width
            )
            frames.append(img)
    
    # Plot probability
    plt.figure(figsize=(10, 6))
    plt.plot(times, probabilities)
    plt.xlabel('Time')
    plt.ylabel('Total Probability')
    plt.title('Total Probability Over Time')
    plt.grid(True)
    plt.savefig("probability_over_time.png")
    plt.close()
    
    # Save animation
    if frames:
        print(f"Saving animation to {output_filename}...")
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(total_time / num_frames * 1000),
            loop=0
        )
        print("Animation complete!")
        return output_filename
    else:
        print("No frames were generated. Please check the simulation parameters.")
        return None

if __name__ == "__main__":
    # Example usage
    output_file = create_tunneling_animation(
        output_filename="a_tunnel.gif",
        num_frames=1000,
        spatial_points=2000,
        spatial_range=(-100, 100),
        total_time=5.0,
        barrier_width=32.6,
        V0=150.0,
        transition_width=0.05,
        vis_settings={'width': 1280, 'height': 640},
        n=4,
        x0=-70.0,
        barrier_center_init=30.215
    )
    print(f"Animation saved to: {output_file}")