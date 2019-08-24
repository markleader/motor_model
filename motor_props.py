import numpy as np
import openmdao.api as om
import dymos as dm
from dymos import declare_time, declare_state, declare_parameter

'''
Compute the geometric and thermal properties of the motor housing structure
'''
class MotorProps(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        
    def setup(self):
        nn = self.options['num_nodes']

        # Define the inputs
        self.add_input('x', 
                       desc='shell thickness',
                       val=10.0e-3, units='m')
        self.add_input('ro', desc='shell outer radius', units='m')
        self.add_input('L_s', desc='shell length', units='m')
        self.add_input('L_p', desc='plate charachteristic length', units='m')
        self.add_input('r_p_small', desc='radius of the small holes in the plate', units='m')
        self.add_input('r_p_large', desc='radius of the large hole through the center of the plate', units='m')
        self.add_input('V_air', desc='volume of air within the nacelle', units='m**3')

        self.add_input('rho_alum', desc='density of aluminum', units='ks/m**3')
        self.add_input('rho_air', desc='density of air', units='kg/m**3')
        self.add_input('k', desc='thermal conductivity of aluminum', units='W/m*K') 
        self.add_input('k_air', desc='thermal conductivity of air', units='W/m*K')
        self.add_input('nu_air', desc='kinematic viscosity of air @ 50 C', units='m**2/s')
        self.add_input('Pr', desc='Prandtl number of air @ 300K, 1 bar', units=None)
        self.add_input('h_factor', desc='correction factor for unforced convective heat transfer coefficients', units=None)

        self.add_input('T_s', desc='temperature of the shell', units='K')
        self.add_input('T_p', desc='temperature of the plate', units='K')
        self.add_input('T_a', desc='temperature of the internal air', units='K')
        self.add_input('T_inf', desc='atmospheric temperature', units='K')
        
        self.add_input('u_inf', shape=(nn,),
                       desc='free stream velocity',
                       units='m/s')
        
        # Define the outputs
        self.add_output('R_s', units='K/W',
                        desc='thermal resistivity of the shell')
        self.add_output('R_p', units='K/W',
                        desc='thermal resistivity of the plate')
        self.add_output('h_s', shape=(nn,), units='W/K*m**2',
                        desc='heat transfer coefficient from the shell to the freestream')
        self.add_output('h_p', shape=(nn,), units='W/K*m**2',
                        desc='heat transfer coefficient from the plate to the internal air')
        
        self.add_output('m_s', 
                        desc='mass of the shell', units='kg')
        self.add_output('m_p', 
                        desc='mass of the shell', units='kg')
        self.add_output('m_a', 
                        desc='mass of the shell', units='kg')
        self.add_output('A_s', units='m**2',
                        desc='convective area of the shell')
        self.add_output('A_p', units='m**2',
                        desc='convective area of the plate')

        # Define the partials
        self.declare_partials(of='R_s', wrt='x')
                              # rows=np.zeros(nn),
                              # cols=np.zeros(nn))
        self.declare_partials(of='m_s', wrt='x')
                              # rows=np.zeros(nn),
                              # cols=np.zeros(nn))

    def compute(self, inputs, outputs):

        # Get the geometry inputs
        x = inputs['x']
        ro = inputs['ro']
        L_s = inputs['L_s']
        L_p = inputs['L_p']
        r_p_small = inputs['r_p_small']
        r_p_large = inputs['r_p_large']
        V_air = inputs['V_air']
        
        # Get the material property inputs
        rho_alum = inputs['rho_alum']
        rho_air = inputs['rho_air']
        k = inputs['k']
        k_air = inputs['k_air']
        nu_air = inputs['nu_air']
        Pr = inputs['Pr']
        h_factor = inputs['h_factor']
        
        # Get the temperature inputs
        T_s = inputs['T_s']
        T_p = inputs['T_p']
        T_a = inputs['T_a']
        T_inf = inputs['T_inf']

        # Get the other inputs
        u_inf = inputs['u_inf']
        g = 9.81

        # Compute the convective areas
        A_s = 2.0*np.pi*ro*L_s
        A_p = np.pi*((ro-x)**2 - r_p_large**2 - 3.0*r_p_small**2)

         # Compute the volumes
        V_s = np.pi*L_s*(ro**2 - (ro-x)**2)
        V_p = L_p*A_p
        
        # Compute the convective heat transfer coefficients
        T_ratio = np.abs((T_p/T_a)-1.0)
        Gr_1_4 = ((g*((2.*(ro-x))**3)*T_ratio)**0.25)*(nu_air**(-0.5)) # Grashof number^(1/4)
        h_p = (0.59*Gr_1_4*(Pr**0.25))*k_air/(2.0*(ro-x))
        h_p *= h_factor
        
        if u_inf > 0.0:
            Re = u_inf*L_s/nu_air
            Nu_s = 0.664*(Re**0.5)*(Pr**0.33)
            h_s = h_factor*Nu_s*k_air/L_s
        else:
            Gr = g*(L_s**3)*((T_s/T_inf)-1.0)/nu_air**2 # Grashof number
            Ra = Gr*Pr # Rayleigh number
            Nu_s = 0.405*Ra**0.25 # Nusselt number for the shell
            h_s = Nu_s*k_air/L_s

        outputs['R_s'] = np.log(ro/(ro-x))/(2.0*np.pi*k*L_s)
        outputs['R_p'] = L_p/(A_p*k)
        outputs['h_s'] = h_s
        outputs['h_p'] = h_p
        outputs['m_s'] = rho_alum*V_s
        outputs['m_p'] = rho_alum*V_p
        outputs['m_a'] = rho_air*V_air
        outputs['A_s'] = A_s
        outputs['A_p'] = A_p

    def compute_partials(self, inputs, partials):

        x = inputs['x']
        ro = inputs['ro']
        L_s = inputs['L_s']
        k = inputs['k']
        rho = inputs['rho_alum']
        
        partials['R_s', 'x'] = (1.0/(ro - x))/(2.0*np.pi*k*L_s)
        partials['m_s', 'x'] = 2.0*np.pi*rho*L_s*(ro - x)

if __name__ == '__main__':

    num_nodes = 1

    prob = om.Problem(model=MotorProps(num_nodes=num_nodes))
    model = prob.model

    prob.setup()
    print('Setup model complete')
    prob.run_model()
    print('Run model complete')
    derivs = prob.check_partials(compact_print=True)

    print('done')

