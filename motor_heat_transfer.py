from __future__ import print_function, division, absolute_import
import numpy as np
import openmdao.api as om
import dymos as dm
from dymos import declare_time, declare_state, declare_parameter

'''
Defines the heat transfer ODE for the simplified motor system
'''
class MotorHeatTransferODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        
    def setup(self):
        nn = self.options['num_nodes']

        # Define the state variables as inputs
        self.add_input('T_m', shape=(nn,),
                       desc='temperature', units='K')
        self.add_input('T_s', shape=(nn,),
                       desc='temperature', units='K')
        self.add_input('T_p', shape=(nn,),
                       desc='temperature', units='K')
        self.add_input('T_a', shape=(nn,),
                       desc='temperature', units='K')

        # Define the remaining inputs
        self.add_input('R_s', units='K/W',
                       desc='thermal resistivity of the shell')
        self.add_input('R_p', units='K/W',
                       desc='thermal resistivity of the plate')
        self.add_input('h_s', shape=(nn,), units='W/K*m**2',
                       desc='heat transfer coefficient from the shell to the freestream')
        self.add_input('h_p', shape=(nn,), units='W/K*m**2',
                       desc='heat transfer coefficient from the plate to the internal air')

        self.add_input('m_m', 
                       desc='mass of the shell', units='kg')
        self.add_input('m_s', 
                       desc='mass of the shell', units='kg')
        self.add_input('m_p', 
                       desc='mass of the shell', units='kg')
        self.add_input('m_a', 
                       desc='mass of the shell', units='kg')
        self.add_input('A_s', units='m**2',
                       desc='convective area of the shell')
        self.add_input('A_p', units='m**2',
                       desc='convective area of the plate')
        self.add_input('c_m', units='J/kg*K',
                       desc='specific heat capacity of the motor')
        self.add_input('c', units='J/kg*K',
                       desc='specific heat capacity of the shell and plate')
        self.add_input('cv_a', units='J/kg*K',
                       desc='volumetric specific heat capacity of the nacelle air')
        self.add_input('P_full', units='W',
                       desc='motor power output at 100\%')
        self.add_input('eta_m', units=None,
                       desc='motor efficiency')
        self.add_input('throttle', units=None,
                       desc='throttle percent')
        self.add_input('T_inf', units='K',
                       desc='free stream temperature')
        self.add_input('T0', units='K',
                       desc='temperature offset for the ambient temperature')
        
        # Define the outputs
        self.add_output('Qdot_in', shape=(nn,),
                        desc='input heat transfer rate', units='W')
        self.add_output('Qdot_s', shape=(nn,),
                        desc='motor to shell heat transfer rate',
                        units='W')
        self.add_output('Qdot_p', shape=(nn,),
                        desc='motor to plate heat transfer rate',
                        units='W')
        self.add_output('Qdot_c', shape=(nn,),
                        desc='convection heat transfer rate',
                        units='W')
        self.add_output('Qdot_cs', shape=(nn,),
                        desc='still convection heat transfer rate',
                        units='W')
        
        self.add_output('Tdot_m', shape=(nn,),
                        desc='motor temperature rate', units='K/s')
        self.add_output('Tdot_s', shape=(nn,),
                        desc='shell temperature rate', units='K/s')
        self.add_output('Tdot_p', shape=(nn,),
                        desc='plate temperature rate', units='K/s')
        self.add_output('Tdot_a', shape=(nn,),
                        desc='air temperature rate', units='K/s')

        # Define the partials
        r = np.arange(nn)
        self.declare_partials(of='Qdot_s', wrt=['T_m', 'T_s'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Qdot_s', wrt='R_s',
                              rows=r,
                              cols=np.zeros(nn))
        self.declare_partials(of='Qdot_p', wrt=['T_m', 'T_p'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Qdot_c', wrt='T_s',
                              rows=r,
                              cols=r)
        self.declare_partials(of='Qdot_cs', wrt=['T_p', 'T_a'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Tdot_m', wrt=['Qdot_p', 'Qdot_s'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Tdot_s', wrt=['Qdot_s', 'Qdot_c'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Tdot_s', wrt='m_s',
                              rows=r,
                              cols=np.zeros(nn))
        self.declare_partials(of='Tdot_p', wrt=['Qdot_p', 'Qdot_cs'],
                              rows=r,
                              cols=r)
        self.declare_partials(of='Tdot_a', wrt='Qdot_cs',
                              rows=r,
                              cols=r)

    def compute(self, inputs, outputs):

        # Get the state variables
        T_m = inputs['T_m']
        T_s = inputs['T_s']
        T_p = inputs['T_p']
        T_a = inputs['T_a']
        
        # Get the property inputs
        R_s = inputs['R_s']
        R_p = inputs['R_p']
        h_s = inputs['h_s']
        h_p = inputs['h_p']
        m_m = inputs['m_m']
        m_s = inputs['m_s']
        m_p = inputs['m_p']
        m_a = inputs['m_a']
        A_s = inputs['A_s']
        A_p = inputs['A_p']
        c_m = inputs['c_m']
        c = inputs['c']
        cv_a = inputs['cv_a']
        throttle = inputs['throttle']
        P_full = inputs['P_full']
        eta_m = inputs['eta_m']
        T_inf = inputs['T_inf']
        
        # Compute some values
        P_m = inputs['P_full']*throttle
        T_inf += inputs['T0']
        
        # Solve for the heat transfer rates
        Qdot_in = P_m*(1.0 - eta_m)
        Qdot_s = (T_m - T_s)/R_s
        Qdot_p = (T_m - T_p)/R_p
        Qdot_c = A_s*h_s*(T_s - T_inf)
        Qdot_cs = A_p*h_p*(T_p - T_a)

        # Solve for the temperature rates
        Tdot_m = (Qdot_in - Qdot_s - Qdot_p)/(m_m*c_m)
        Tdot_s = (Qdot_s - Qdot_c)/(m_s*c)
        Tdot_p = (Qdot_p - Qdot_cs)/(m_p*c)
        Tdot_a = Qdot_cs/(m_a*cv_a)

        # Set the outputs
        outputs['Qdot_in'] = Qdot_in
        outputs['Qdot_s'] = Qdot_s
        outputs['Qdot_p'] = Qdot_p
        outputs['Qdot_c'] = Qdot_c
        outputs['Qdot_cs'] = Qdot_cs

        outputs['Tdot_m'] = Tdot_m
        outputs['Tdot_s'] = Tdot_s
        outputs['Tdot_p'] = Tdot_p
        outputs['Tdot_a'] = Tdot_a

    def compute_partials(self, inputs, partials):

        # Get some inputs
        T_m = inputs['T_m']
        T_s = inputs['T_s']
        T_inf = inputs['T_inf']
        R_s = inputs['R_s']
        R_p = inputs['R_p']
        h_s = inputs['h_s']
        h_p = inputs['h_p']
        A_s = inputs['A_s']
        A_p = inputs['A_p']
        m_m = inputs['m_m']
        m_s = inputs['m_s']
        m_p = inputs['m_p']
        m_a = inputs['m_a']
        c = inputs['c']
        cv_a = inputs['cv_a']
        Qdot_s = (T_m - T_s)/R_s
        Qdot_c = A_s*h_s*(T_s - T_inf)
        
        # Compute the partials
        partials['Qdot_s', 'T_m'] = 1.0/R_s
        partials['Qdot_s', 'T_s'] = -1.0/R_s
        partials['Qdot_s', 'R_s'] = -(T_m - T_s)/R_s**2

        partials['Qdot_p', 'T_m'] = 1.0/R_p
        partials['Qdot_p', 'T_p'] = -1.0/R_p
        
        partials['Qdot_c', 'T_s'] = A_s*h_s

        partials['Qdot_cs', 'T_p'] = A_p*h_p
        partials['Qdot_cs', 'T_a'] = -A_p*h_p

        partials['Tdot_m', 'Qdot_p'] = -1.0/(m_m*c)
        partials['Tdot_m', 'Qdot_s'] = -1.0/(m_m*c)

        partials['Tdot_s', 'Qdot_s'] = 1.0/(m_s*c)
        partials['Tdot_s', 'Qdot_c'] = -1.0/(m_s*c)
        partials['Tdot_s', 'm_s'] = -(Qdot_s - Qdot_c)/(c*m_s**2)

        partials['Tdot_p', 'Qdot_p'] = 1.0/(m_p*c)
        partials['Tdot_p', 'Qdot_cs'] = -1.0/(m_p*c)

        partials['Tdot_a', 'Qdot_cs'] = 1.0/(m_a*cv_a)

if __name__ == '__main__':

    num_nodes = 1

    prob = om.Problem(model=MotorHeatTransferODE(num_nodes=num_nodes))
    model = prob.model

    prob.setup()

    prob.run_model()

    derivs = prob.check_partials(compact_print=True)

    print('done')
