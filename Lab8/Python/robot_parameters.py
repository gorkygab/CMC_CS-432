"""Robot parameters"""

import numpy as np
import farms_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        
        drive = parameters.drive
        
        if drive >= 1 and drive <= 5:
            self.freqs[0:20] = (0.4+(drive*0.1)) * np.ones(20)
        else:
            self.freqs[0:20] = np.zeros(20)
            
        
        if drive >= 1 and drive <= 3:
            self.freqs[20:24] = drive*0.15 * np.ones(4)
        else:
            self.freqs[20:24] = np.zeros(4)
            

    def set_coupling_weights(self, parameters):
        """Set coupling weights"""
        
        for i in range(10):
            if i != 0:
                self.coupling_weights[i,i-1] = 10
            self.coupling_weights[i,i+1] = 10
            self.coupling_weights[i,i+10] = 10  
            
        for i in range(10):
            self.coupling_weights[10+i,i+9] = 10
            if i != 9:
                self.coupling_weights[10+i,i+11] = 10
            self.coupling_weights[10+i,i] = 10
            
            
        for i in range(5):
            self.coupling_weights[20,i] = 30
            self.coupling_weights[21,i+5] = 30
            self.coupling_weights[22,i+10] = 30
            self.coupling_weights[23,i+15] = 30 
            
        self.coupling_weights[20,21] = 10
        self.coupling_weights[20,22] = 10
        
        self.coupling_weights[21,20] = 10
        self.coupling_weights[21,23] = 10
        
        self.coupling_weights[22,20] = 10
        self.coupling_weights[22,23] = 10
        
        self.coupling_weights[23,21] = 10
        self.coupling_weights[23,22] = 10

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        if parameters.phase_lag is None:
            phase_lag = 2*np.pi/10
        else:
            phase_lag = parameters.phase_lag/parameters.n_body_joints
                
        limb_lag = parameters.limb_lag
        
        for i in range(10):
            if i != 0:
                self.phase_bias[i,i-1] = -phase_lag
            self.phase_bias[i,i+1] = phase_lag
            self.phase_bias[i,i+10] = np.pi 
            self.phase_bias[20+int(i/5),i] = limb_lag
            
        for i in range(10):
            self.phase_bias[10+i,i+9] = -phase_lag
            if i != 9:
                self.phase_bias[10+i,i+11] = phase_lag
            self.phase_bias[10+i,i] = np.pi 
            self.phase_bias[22+int(i/5),10+i] = limb_lag
            
            
        self.phase_bias[20,21] = np.pi
        self.phase_bias[20,22] = np.pi
        
        self.phase_bias[21,20] = np.pi
        self.phase_bias[21,23] = np.pi
        
        self.phase_bias[22,20] = np.pi
        self.phase_bias[22,23] = np.pi
        
        self.phase_bias[23,21] = np.pi
        self.phase_bias[23,22] = np.pi

    def set_amplitudes_rate(self, parameters):
        """Set amplitude rates"""
        self.rates[0:20] = 1 * np.ones(20)
        self.rates[20:24] = 1 * np.ones(4)

    def set_nominal_amplitudes(self, parameters):
        """Set nominal amplitudes"""
        turn = parameters.turn
        
        turnVect = np.zeros(20)
        if turn < 0:
            turnVect = np.reshape([np.ones(10) * np.abs(turn), np.zeros(10)], (1,20))
        if turn > 0:
            turnVect = np.reshape([np.zeros(10), np.ones(10) * np.abs(turn)], (1,20))
        
        gradvect = np.linspace(parameters.rhead, parameters.rtail, self.n_body_joints)
        gradvect = np.reshape([gradvect, gradvect], (1,20))
    
        drive = parameters.drive
        if parameters.amplitudes is None:
            amplitude = 1
        else:
            amplitude = parameters.amplitudes
        
        if drive >= 1 and drive <= 5:
            self.nominal_amplitudes[0:20] = (drive*0.05)* (np.ones(20) - turnVect) * amplitude
        else:
            self.nominal_amplitudes[0:20] = np.zeros(20)
            
        self.nominal_amplitudes[0:20] = np.multiply(self.nominal_amplitudes[0:20],gradvect)
        
        #Limbs
        if drive >= 1 and drive <= 3:
            self.nominal_amplitudes[20:24] = (0.1+(drive*0.37))* np.ones(4) * amplitude
        else:
            self.nominal_amplitudes[20:24] = np.zeros(4)

