# modules/wave_packet_tunneling/main.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from .wavefunction import WaveFunction
from .potential import PotentialBarrier
from .evolution import WaveFunctionEvolution
from .visualization import Visualizer

class WavePacketMenu:
    def __init__(self, main_gui):
        self.main_gui = main_gui
        self.output_text = ""
        self.sim_params = {
            'output_filename': 'quantum_tunneling.gif',
            'num_frames': 120,
            'spatial_points': 2000,
            'spatial_range': (-100, 100),
            'total_time': 6.0,
            'barrier_width': 32.6,
            'V0': 150.0,
            'transition_width': 0.05,
            'vis_settings': {'width': 1280, 'height': 640},
            'n': 4,
            'x0': -70.0,
            'barrier_center_init': 30.215
        }
        self.menu_state = 'main'
        self.param_input_state = None

    def get_menu_text(self):
        if self.menu_state == 'main':
            return (
                "Wave Packet Tunneling Simulator\n"
                "================================\n\n"
                "1. Run simulation\n"
                "2. Modify parameters\n"
                "3. View current parameters\n"
                "4. Reset to defaults\n"
                "5. Exit\n\n"
                "h - home\n"
                "b - back\n\n"
                "Enter your choice: "
            )
        elif self.menu_state == 'params':
            return (
                "Parameter Configuration\n"
                "======================\n\n"
                "1. Initial position (x0): {:.1f}\n"
                "2. Barrier width: {:.1f}\n"
                "3. Barrier height (V0): {:.1f}\n"
                "4. Energy levels (n): {}\n"
                "5. Barrier center: {:.3f}\n"
                "6. Number of frames: {}\n"
                "7. Total simulation time: {:.1f}\n"
                "8. Return to main menu\n\n"
                "Enter parameter number to modify: "
            ).format(
                self.sim_params['x0'],
                self.sim_params['barrier_width'],
                self.sim_params['V0'],
                self.sim_params['n'],
                self.sim_params['barrier_center_init'],
                self.sim_params['num_frames'],
                self.sim_params['total_time']
            )

    def get_output(self):
        return self.output_text

    def execute_command(self, command):
        self.output_text = ""
        
        if command.lower() == 'h':
            self.menu_state = 'main'
            self.param_input_state = None
            return
        elif command.lower() == 'b':
            if self.param_input_state:
                self.param_input_state = None
            elif self.menu_state == 'params':
                self.menu_state = 'main'
            return

        if self.param_input_state:
            self._handle_param_input(command)
            return

        try:
            choice = int(command)
            if self.menu_state == 'main':
                self._handle_main_menu(choice)
            elif self.menu_state == 'params':
                self._handle_params_menu(choice)
        except ValueError:
            self.output_text = "Invalid input. Please enter a number."

    def _handle_main_menu(self, choice):
        if choice == 1:
            self._run_simulation()
        elif choice == 2:
            self.menu_state = 'params'
        elif choice == 3:
            self._display_current_params()
        elif choice == 4:
            self._reset_params()
        elif choice == 5:
            self.main_gui.application.exit()
        else:
            self.output_text = "Invalid choice. Please select 1-5."

    def _handle_params_menu(self, choice):
        if choice == 8:
            self.menu_state = 'main'
            return

        param_map = {
            1: ('x0', 'Enter new initial position (-100.0 to 0.0): ', float, -100.0, 0.0),
            2: ('barrier_width', 'Enter new barrier width (0.1-50.0): ', float, 0.1, 50.0),
            3: ('V0', 'Enter new barrier height (1.0-200.0): ', float, 1.0, 200.0),
            4: ('n', 'Enter new number of energy levels (1-10): ', int, 1, 10),
            5: ('barrier_center_init', 'Enter new barrier center (-50.0 to 50.0): ', float, -50.0, 50.0),
            6: ('num_frames', 'Enter new number of frames (30-240): ', int, 30, 240),
            7: ('total_time', 'Enter new simulation time (1.0-20.0): ', float, 1.0, 20.0)
        }

        if choice in param_map:
            param_name, prompt, param_type, min_val, max_val = param_map[choice]
            self.param_input_state = {
                'param_name': param_name,
                'prompt': prompt,
                'type': param_type,
                'min_val': min_val,
                'max_val': max_val
            }
            self.output_text = prompt
        else:
            self.output_text = "Invalid choice. Please select 1-8."

    def _handle_param_input(self, input_value):
        try:
            value = self.param_input_state['type'](input_value)
            if (value >= self.param_input_state['min_val'] and 
                value <= self.param_input_state['max_val']):
                self.sim_params[self.param_input_state['param_name']] = value
                self.output_text = f"Parameter updated successfully."
                self.param_input_state = None
            else:
                self.output_text = (f"Value must be between "
                                  f"{self.param_input_state['min_val']} and "
                                  f"{self.param_input_state['max_val']}")
        except ValueError:
            self.output_text = "Invalid input. Please enter a number."

    def _run_simulation(self):
        self.output_text = "Starting simulation...\n"
        try:
            output_file = self._create_tunneling_animation()
            self.output_text += f"Simulation completed! Animation saved to: {output_file}"
        except Exception as e:
            self.output_text += f"Error during simulation: {str(e)}"

    def _create_tunneling_animation(self):
        """
        Create an animation of quantum tunneling with current parameters.
        """
        # Initialize spatial grid
        x = np.linspace(self.sim_params['spatial_range'][0], 
                       self.sim_params['spatial_range'][1], 
                       self.sim_params['spatial_points'])
        dx = x[1] - x[0]
        
        # Initialize components
        psi = WaveFunction.initialize_wavefunction(x, self.sim_params['n'], self.sim_params['x0'])
        V_total = PotentialBarrier.create_total_potential(
            x, self.sim_params['barrier_center_init'], 
            self.sim_params['V0'], self.sim_params['barrier_width'], 
            self.sim_params['transition_width']
        )
        visualizer = Visualizer(self.sim_params['vis_settings'])
        
        # Calculate time step
        max_k = np.max(np.abs(2 * np.pi * np.fft.fftfreq(self.sim_params['spatial_points'], d=dx)))
        dt = 0.05 / (max_k**2 / 2)
        
        # Setup time evolution
        num_time_steps = int(self.sim_params['total_time'] / dt) + 1
        times = np.linspace(0, self.sim_params['total_time'], num_time_steps)
        frame_indices = np.linspace(0, num_time_steps - 1, self.sim_params['num_frames']).astype(int)
        
        frames = []
        probabilities = []
        
        self.output_text += "Generating frames...\n"
        for i in range(num_time_steps):
            # Evolve wave function
            psi = WaveFunctionEvolution.evolve_wavefunction(psi, x, dt, V_total)
            
            # Record probability
            total_prob = WaveFunction.compute_total_probability(psi, dx)
            probabilities.append(total_prob)
            
            # Capture frame if needed
            if i in frame_indices:
                img = visualizer.create_datashader_frame(
                    psi, x,
                    barrier_center=self.sim_params['barrier_center_init'],
                    barrier_width=self.sim_params['barrier_width']
                )
                frames.append(img)
        
        # Save probability plot
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
            self.output_text += f"Saving animation to {self.sim_params['output_filename']}...\n"
            frames[0].save(
                self.sim_params['output_filename'],
                save_all=True,
                append_images=frames[1:],
                duration=int(self.sim_params['total_time'] / self.sim_params['num_frames'] * 1000),
                loop=0
            )
            return self.sim_params['output_filename']
        else:
            raise RuntimeError("No frames were generated. Please check the simulation parameters.")

    def _display_current_params(self):
        self.output_text = "Current Parameters:\n\n"
        exclude_params = {'vis_settings', 'spatial_range', 'spatial_points', 'transition_width'}
        for key, value in self.sim_params.items():
            if key not in exclude_params:
                self.output_text += f"{key}: {value}\n"

    def _reset_params(self):
        self.sim_params = {
            'output_filename': 'quantum_tunneling.gif',
            'num_frames': 120,
            'spatial_points': 2000,
            'spatial_range': (-100, 100),
            'total_time': 6.0,
            'barrier_width': 32.6,
            'V0': 150.0,
            'transition_width': 0.05,
            'vis_settings': {'width': 1280, 'height': 640},
            'n': 4,
            'x0': -70.0,
            'barrier_center_init': 30.215
        }
        self.output_text = "Parameters reset to defaults."