""" Lab 5 - Exercise 1 """

import matplotlib.pyplot as plt
import numpy as np

import farms_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
    parameters = MuscleParameters()
    pylog.warning("Loading default muscle parameters")
    pylog.info(parameters.showParameters())
    pylog.info("Use the parameters object to change the muscle parameters")

    # Change the fiber length property l_opt
   
    lopt = 0.35
    parameters.l_opt = lopt
    
    # Create muscle object
    muscle = Muscle(parameters)

    pylog.warning("Isometric muscle contraction to be completed")

    # Instatiate isometric muscle system
    sys = IsometricMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single muscle stretch
    muscle_stretch = 0.8

    # Evalute for a single muscle stimulation
    muscle_stimulation = [0.2, 0.4, 0.6, 0.8, 1]
    stim = 0.7
    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001

    time = np.arange(t_start, t_stop, time_step)


    #for stim in muscle_stimulation:
    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           stimulation=stim,
                           muscle_length=muscle_stretch)


    
    #pylog.info(result)
    # Plotting
    plt.figure('Isometric muscle experiment 1b')
    
    '''
    plt.plot(result.l_ce, result.active_force)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Contractile element length [m]')
    plt.ylabel('Active force [N]')
    plt.plot(result.l_ce, result.passive_force)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Contractile element length [m]')
    plt.ylabel('Passive force [N]')
    '''
    plt.plot(result.l_ce, result.passive_force + result.active_force)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Contractile element length [m]')
    plt.ylabel('Total force [N]')
    plt.xlim(0.1,0.5)
    plt.ylim(50, 1800)
    plt.grid()
        
    


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    mass_parameters.mass = 0
    print(mass_parameters.showParameters())
    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.l_opt # To get the muscle optimal length

    # Evalute for a single load
    load = np.arange(50/9.81, 3000/9.81, 5)

    # Evalute for a single muscle stimulation

    # Set the initial condition
    x0 = [0.0, sys.muscle.l_opt,
          sys.muscle.l_opt + sys.muscle.l_slack, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.4
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    
    plt.figure('Isotonic muscle experiment')
    muscle_stimulation = [0.2, 0.4, 0.6, 0.8, 1]
    for j in range(len(muscle_stimulation)):
        vel = np.zeros(len(load))
        
        for i in range(len(load)):
            result = sys.integrate(x0=x0,
                                time=time,
                                time_step=time_step,
                                time_stabilize=time_stabilize,
                                stimulation=muscle_stimulation[j],
                                load=load[i]
                                )
            if(result.l_mtu[len(result.l_mtu)-1] < (sys.muscle.l_opt + sys.muscle.l_slack)):
                vel[i] = min(result.v_ce)
            else:
                vel[i] = max(result.v_ce)              
        
        #pylog.info(result)
        # Plotting
        plt.plot(vel, load*9.81, label='Stimulation {}'.format(muscle_stimulation[j]))
        plt.axvline(0, 0, 3000, color = 'red')

        plt.title('Isotonic muscle experiment')
        plt.xlabel('Muscle velocity [m/s]')
        plt.ylabel('Tension [N]')
    
    
    
    
    ax = plt.gca()
    ax.invert_xaxis()
    ax.axvspan(0.5, 0.0, alpha=0.2, color='red')
    ax.axvspan(0.0, -1.18, alpha=0.2, color='blue')
    ax.legend()
    plt.grid()


def exercise1():
    #exercise1a()
    exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    #exercise1()
    exercise1d()

