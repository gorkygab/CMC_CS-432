"""Exercise 8g"""

import pickle
import numpy as np
import matplotlib.animation as manimation
# from simulation import simulation  # You can also implement another function
from simulation_parameters import SimulationParameters
from simulation import simulation
import farms_pylog as pylog


def exercise_8g(timestep):
    """Exercise 8g"""
    parameter_set = [
        SimulationParameters(
            duration=80,  # Simulation duration in [s]
            timestep=timestep,  # Simulation timestep in [s]
            spawn_position=[0, 0, 0.1],  # Robot position in [m]
            spawn_orientation=[0, 0, np.pi],  # Orientation in Euler angles [rad]
            drive=4,  # An example of parameter part of the grid search
            amplitudes=0.89,  # From 0 to 2 with 5 iterations [0,0.5,1,1.5,2]
            phase_lag=2*np.pi,  # From 0 to 2 with 5 iterations [0,0.5,1,1.5,2]
            turn=0,  # Another example
            limb_lag = -np.pi/2,
            rhead = 1,
            rtail = 1,
            # ...
        )
        # for amplitudes in ...
        # for ...
        ]
    
    # Grid search
    for simulation_i, sim_parameters in enumerate(parameter_set):
        filename = './logs/8g/simulation_{}.{}'
        sim, data = simulation(
            sim_parameters=sim_parameters,  # Simulation parameters, see above
            arena='amphibious',  # Can also be 'ground' or 'amphibious'
            #fast=True,  # For fast mode (not real-time)
            # headless=True,  # For headless mode (No GUI, could be faster)
            # record=True,  # Record video, see below for saving
            # video_distance=1.5,  # Set distance of camera to robot
            # video_yaw=0,  # Set camera yaw for recording
            # video_pitch=-45,  # Set camera pitch for recording
        )
        # Log robot data
        data.to_file(filename.format(simulation_i, 'h5'), sim.iteration)
        # Log simulation parameters
        with open(filename.format(simulation_i, 'pickle'), 'wb') as param_file:
            pickle.dump(sim_parameters, param_file)
        # Save video
        if sim.options.record:
            if 'ffmpeg' in manimation.writers.avail:
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.mp4',
                    iteration=sim.iteration,
                    writer='ffmpeg',
                )
            elif 'html' in manimation.writers.avail:
                # FFmpeg might not be installed, use html instead
                sim.interface.video.save(
                    filename='salamandra_robotica_simulation.html',
                    iteration=sim.iteration,
                    writer='html',
                )
            else:
                pylog.error('No known writers, maybe you can use: {}'.format(
                    manimation.writers.avail
                ))
    pass


if __name__ == '__main__':
    exercise_8g(timestep=1e-2)

