"""Oscillator network ODE"""

import numpy as np

from scipy.integrate import ode
from robot_parameters import RobotParameters


def network_ode(_time, state, robot_parameters):
    """Network_ODE

    Parameters
    ----------
    _time: <float>
        Time
    state: <np.array>
        ODE states at time _time
    robot_parameters: <RobotParameters>
        Instance of RobotParameters

    Return
    ------
    :<np.array>
        Returns derivative of state (phases and amplitudes)
    """
    n_oscillators = robot_parameters.n_oscillators
    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2*n_oscillators]
    weights = robot_parameters.coupling_weights
    phase_bias = robot_parameters.phase_bias
    nominal_amp = robot_parameters.nominal_amplitudes
    rates = robot_parameters.rates
    freqs = robot_parameters.freqs
    
    
    #COMMENTED CODE BELOW IS USED FOR LINEAR DRIVE FOR 8a_3
    '''
    drive = _time*6/30
        
    if drive >= 1 and drive <= 5:
        nominal_amp[0:20] = (0.25+(drive*0.05))* np.ones(20)
    else:
        nominal_amp[0:20] = np.zeros(20)
        
    
    if drive >= 1 and drive <= 3:
        nominal_amp[20:24] = (0.2+(drive*0.1))* np.ones(4)
    else:
        nominal_amp[20:24] = np.zeros(4)

        
    if drive >= 1 and drive <= 5:
        freqs[0:20] = (0.2+(drive*0.3)) * np.ones(20)
    else:
        freqs[0:20] = np.zeros(20)
        
    
    if drive >= 1 and drive <= 3:
        freqs[20:24] = drive*0.2 * np.ones(4)
    else:
        freqs[20:24] = np.zeros(4)'''
            
    
    
    phasesdot = np.zeros(24)
    amplitudesdot = np.zeros(24)
    # Implement equation here
    for i in range(24):
        sumElements = robot_parameters.coupling_weights[:,i] * amplitudes * np.sin(phases - np.ones(24)*phases[i] - robot_parameters.phase_bias[:,i])
        phasesdot[i] = 2*np.pi*freqs[i] + np.sum(sumElements)
        amplitudesdot[i] = robot_parameters.rates[i]*(nominal_amp[i] - amplitudes[i])
    
    return np.concatenate([phasesdot, amplitudesdot])


def motor_output(phases, amplitudes, iteration=None):
    """Motor output.

    Parameters
    ----------
    phases: <np.array>
        Phases of the oscillator
    amplitudes: <np.array>
        Amplitudes of the oscillator

    Returns
    -------
    : <np.array>
        Motor outputs for joint in the system.
    """
    # Implement equation here
    body = np.zeros(10)
    for i in range(10):
        body[i] = amplitudes[i]*(1+np.cos(phases[i])) - amplitudes[i+10]*(1+np.cos(phases[i+10]))
    
    # Implement equation here        
    limbs = np.zeros(4)
    
    for i in range(4):
        limbs[i] = amplitudes[i+20]*(1+np.cos(phases[i+20]))
    
    return np.concatenate([body, limbs])


class RobotState(np.ndarray):
    """Robot state"""

    def __init__(self, *_0, **_1):
        super(RobotState, self).__init__()
        self[:] = 0.0

    @classmethod
    def salamandra_robotica_2(cls, n_iterations):
        """State of Salamandra robotica 2"""
        shape = (n_iterations, 2*24)
        return cls(
            shape,
            dtype=np.float64,
            buffer=np.zeros(shape)
        )

    def phases(self, iteration=None):
        """Oscillator phases"""
        return self[iteration, :24] if iteration is not None else self[:, :24]

    def set_phases(self, iteration, value):
        """Set phases"""
        self[iteration, :24] = value

    def set_phases_left(self, iteration, value):
        """Set body phases on left side"""
        self[iteration, :10] = value

    def set_phases_right(self, iteration, value):
        """Set body phases on right side"""
        self[iteration, 10:20] = value

    def set_phases_legs(self, iteration, value):
        """Set leg phases"""
        self[iteration, 20:24] = value

    def amplitudes(self, iteration=None):
        """Oscillator amplitudes"""
        return self[iteration, 24:] if iteration is not None else self[:, 24:]

    def set_amplitudes(self, iteration, value):
        """Set amplitudes"""
        self[iteration, 24:] = value


class SalamandraNetwork:
    """Salamandra oscillator network"""

    def __init__(self, sim_parameters, n_iterations):
        super(SalamandraNetwork, self).__init__()
        # States
        self.state = RobotState.salamandra_robotica_2(n_iterations)
        # Parameters
        self.robot_parameters = RobotParameters(sim_parameters)
        # Set initial state
        # Replace your oscillator phases here
        self.state.set_phases(
            iteration=0,
            value=1e-4*np.random.ranf(self.robot_parameters.n_oscillators),
        )
        # Set solver
        self.solver = ode(f=network_ode)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(y=self.state[0], t=0.0)

    def step(self, iteration, time, timestep):
        """Step"""
        self.solver.set_f_params(self.robot_parameters)
        self.state[iteration+1, :] = self.solver.integrate(time+timestep)

    def outputs(self, iteration=None):
        """Oscillator outputs"""
        # Implement equation here
        return np.zeros(14)

    def get_motor_position_output(self, iteration=None):
        """Get motor position"""
        return motor_output(
            self.state.phases(iteration=iteration),
            self.state.amplitudes(iteration=iteration),
            iteration=iteration,
        )

