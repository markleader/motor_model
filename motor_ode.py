import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import dymos as dm
from dymos.utils.lgl import lgl
from motor_props import MotorProps
from motor_heat_transfer import MotorHeatTransferODE
from atmosphere import USatm1976Comp

# Define the inputs and outputs of the motor properties subsystem
props_inputs = ['x', 'ro', 'L_s', 'L_p', 'r_p_small',
                'r_p_large', 'V_air', 'rho_alum', 'rho_air',
                'k', 'k_air', 'nu_air', 'Pr', 'h_factor',
                'T_s', 'T_p', 'T_a', 'T_inf', 'u_inf']
props_outputs = ['R_s','R_p', 'h_s', 'h_p', 'm_s',
                 'm_p', 'm_a', 'A_s', 'A_p']
# Define the inputs and outputs of the heat transfer subsystem
ht_inputs = ['T_m', 'T_s', 'T_p', 'T_a',
             'R_s', 'R_p', 'h_s', 'h_p', 'm_m', 'm_s',
             'm_p', 'm_a', 'A_s', 'A_p', 'c_m', 'c', 'cv_a',
             'P_full', 'eta_m', 'throttle', 'T_inf', 'T0']
ht_outputs = ['Qdot_in','Qdot_s', 'Qdot_p','Qdot_c',
              'Qdot_cs','Tdot_m', 'Tdot_s','Tdot_p',
              'Tdot_a']

class MotorTempODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Add the motor property subsystem
        self.add_subsystem(name='motor_props',
                           subsys=MotorProps(num_nodes=nn),
                           promotes_inputs=props_inputs,
                           promotes_outputs=props_outputs)
        
        # Add the heat transfer subsystem
        self.add_subsystem(name='heat_transfer',
                           subsys=MotorHeatTransferODE(num_nodes=nn),
                           promotes_inputs=ht_inputs,
                           promotes_outputs=ht_outputs)

        # Add the altitude model
        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'])

        # Connect the temperatures
        self.connect('motor_props.T_s', 'heat_transfer.T_s')
        #self.connect('motor_props.T_p', 'heat_transfer.T_p')
        #self.connect('motor_props.T_a', 'heat_transfer.T_a')
        #self.connect('atmos.temp', ('heat_transfer.T_inf', 'motor_props.T_inf'))

        # Connect the thermal properties
        # self.connect('motor_props.R_s', 'heat_transfer.R_s')
        #self.connect('motor_props.R_p', 'heat_transfer.R_p')
        #self.connect('motor_props.h_s', 'heat_transfer.h_s')
        #self.connect('motor_props.h_p', 'heat_transfer.h_p')
        #self.connect('motor_props.m_s', 'heat_transfer.m_s')
        #self.connect('motor_props.m_p', 'heat_transfer.m_p')
        #self.connect('motor_props.m_a', 'heat_transfer.m_a')
        #self.connect('motor_props.A_s', 'heat_transfer.A_s')
        #self.connect('motor_props.A_P', 'heat_transfer.A_p')
        
