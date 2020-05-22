"""Plot results"""

import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from save_figures import save_figures
from parse_args import save_plots
from salamandra_simulation.data import AnimatData


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True):
    """Main"""
    # Load data
    dataquantity = 100
    
    speed = np.zeros(dataquantity)
    energy = np.zeros(dataquantity)
    amplitudes = np.zeros(dataquantity)
    phase_lag = np.zeros(dataquantity)
    rhead = np.zeros(dataquantity)
    rtail = np.zeros(dataquantity)
    
    for i in range(dataquantity):  # Inner loop should be phase and outer loop should be amplitude
        data = AnimatData.from_file('logs/8c/simulation_{}.h5'.format(i), 2*14)
        with open('logs/8c/simulation_{}.pickle'.format(i), 'rb') as param_file:
            parameters = pickle.load(param_file)
        times = data.times
        timestep = times[1] - times[0]  # Or parameters.timestep
        amplitudes[i] = parameters.amplitudes
        phase_lag[i] = parameters.phase_lag
        rhead[i] = parameters.rhead
        rtail[i] = parameters.rtail
        osc_phases = data.state.phases_all()
        osc_amplitudes = data.state.amplitudes_all()
        links_positions = np.asarray(data.sensors.gps.urdf_positions())
        head_positions = links_positions[:, 0, :]
        tail_positions = links_positions[:, 10, :]
        joints_positions = data.sensors.proprioception.positions_all()
        joints_velocities = data.sensors.proprioception.velocities_all()
        joints_torques = data.sensors.proprioception.motor_torques()
        speed[i] = head_positions[-1,0]/times[-1]
        energy[i] = np.sum(np.multiply(joints_velocities, joints_torques))
    # Notes:
    # For the gps arrays: positions[iteration, link_id, xyz]
    # For the positions arrays: positions[iteration, xyz]
    # For the joints arrays: positions[iteration, joint]
        
        
        
        
        
        
        
    '''
    #FOR EXERCICE 8B 
    result = np.zeros((dataquantity,3))
    result[:,0] = amplitudes
    result[:,1] = phase_lag
    
    plt.figure("Speed")
    result[:,2] = speed
    plot_2d(result,["Amplitude", "Phase lag", "Speed"]) 
    
    plt.figure("Energy")
    result[:,2] = energy
    plot_2d(result,["Amplitude", "Phase lag", "Energy"])
      '''
    
       
    #FOR EXERCISE 8C
    result = np.zeros((dataquantity,3))
    result[:,0] = rhead
    result[:,1] = rtail
    
    plt.figure("Speed")
    result[:,2] = speed
    plot_2d(result,["Head amplitude", "Tail amplitude", "Speed"]) 
    
    plt.figure("Energy")
    result[:,2] = energy
    plot_2d(result,["Head amplitude", "Tail amplitude", "Energy"])
    
    '''
    # Plt data
    plt.figure("Positions")
    plot_positions(times, head_positions)
    plt.figure("Trajectory")
    plot_trajectory(head_positions)
    
    plt.figure("Joint positions")
    plot_positions(times, joints_positions)
    '''
    

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

