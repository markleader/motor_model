import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import dymos as dm
from dymos.utils.lgl import lgl
from motor_ode import MotorTempODE

'''
Define the mission segments
'''
# time at the begining of each segment
t_seg = np.array([0.0, 30.0, 60.0, 70.0,
                  160.0, 700.0, 1000.0, 1450.0,
                  1630.0, 1720.0, 1810.0])

# duration of each segment
dt_seg = np.array([30.0, 30.0, 10.0, 90.0,
                   540.0, 300.0, 450.0, 180.0,
                   90.0, 90.0, 180.0])

# fraction of total power for each segment
p_seg = np.array([1.0, 0.0, 1.0, 0.75,
                  0.0, 0.0, 0.0, 0.75,
                  0.75, 0.0, 0.75, 0.75])

# start and end velocity of each segment
v_seg = np.array([0.0, 0.0, 0.0, 39.92,
                  54.02, 69.45, 69.45, 69.45,
                  54.02, 32.92, 54.02, 54.02, 32.92])

# start and end altitude of each segment
h_seg = np.array([0.0, 0.0, 0.0, 0.0,
                  457.0, 1744.0, 1744.0, 457.0,
                  457.0, 457.0, 457.0, 457.0])
h_seg += 690.0 # add ground level


# Create the problem
prob = om.Problem()

opt = prob.driver = om.ScipyOptimizeDriver()
opt.declare_coloring()
opt.options['optimizer'] = 'SLSQP'

# Set up the segments based on the type of mission
num_seg = 11
    
seg_ends, _ = lgl(num_seg + 1)
traj = prob.model.add_subsystem('traj', dm.Trajectory())

'''
Set the values of the design parameters
'''
# Geometric parameters
# --------------------
traj.add_design_parameter('x', val=10e-3, lower=1e-3, upper=80e-3,
                          units='m', opt=True, dynamic=False)
# targets={'traj_p1':'x', 'traj_p2':'x', 'traj_p3':'x',
#                                    'traj_p4':'x', 'traj_p5':'x', 'traj_p6':'x',
#                                    'traj_p7':'x', 'traj_p8':'x', 'traj_p9':'x',
#                                    'traj_p10':'x', 'traj_p11':'x'}
traj.add_design_parameter('ro', val=80.78e-3, units='m', opt=False, dynamic=False)
# targets={'traj_p1':'ro', 'traj_p2':'ro', 'traj_p3':'ro',
#                                    'traj_p4':'ro', 'traj_p5':'ro', 'traj_p6':'ro',
#                                    'traj_p7':'ro', 'traj_p8':'ro', 'traj_p9':'ro',
#                                    'traj_p10':'ro', 'traj_p11':'ro'}
traj.add_design_parameter('L_s', val=60.71e-3, units='m', opt=False, dynamic=False)
traj.add_design_parameter('L_p', val=6.35e-3, units='m', opt=False, dynamic=False)
traj.add_design_parameter('r_p_small', val=4.08e-3, units='m', opt=False, dynamic=False)
traj.add_design_parameter('r_p_large', val=9.91e-3, units='m', opt=False, dynamic=False)
traj.add_design_parameter('V_air', val=1.98466e-3, units='m**3', opt=False, dynamic=False)

# Motor
traj.add_design_parameter('m_m', val=2.722, opt=False, dynamic=False,
                          desc='mass of the mo', units='kg')
traj.add_design_parameter('eta_m', val=0.96, opt=False, dynamic=False,
                          desc='motor efficiency', units=None)
traj.add_design_parameter('P_full', val=10.3e3, opt=False, dynamic=False,
                          desc='full motor power output', units='W')

# General material properties
# ---------------------------
# Material properties of 6061-T6 aluminum
# http://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MA6061T6
traj.add_design_parameter('rho_alum', val=2600.0, units='kg/m**3', opt=False, dynamic=False)
traj.add_design_parameter('k', val=167.0, units='W/m*K', opt=False, dynamic=False)
traj.add_design_parameter('c_m', val=0.896e3, units='J/kg*K', opt=False, dynamic=False,
                          desc='specific heat capacity of the motor')
traj.add_design_parameter('c', val=0.896e3, units='J/kg*K', opt=False, dynamic=False,
                          desc='specific heat capacity of the motor housing')

# Air properties
traj.add_design_parameter('Pr', val=0.702, opt=False, dynamic=False, units=None,
                          desc='Prandtl number of air, at 300 K, 1 bar')
traj.add_design_parameter('k_air', val=28.8e-3, opt=False, dynamic=False,
                          desc='thermal conductivity of air @ 60 C', units='W/m*K')
traj.add_design_parameter('nu_air', val=17.88e-6, units='m**2/s', opt=False, dynamic=False,
                          desc='kinematic viscosity of air @ 50 C')
traj.add_design_parameter('cv_a', val=0.718e3, units='J/kg*K', opt=False, dynamic=False,
                          desc='volumetric specific heat capacity of air')
traj.add_design_parameter('rho_air', val=1.146, units='kg/m**3', opt=False, dynamic=False,
                          desc='density of air (@ 35 C, 1 atm)')
traj.add_design_parameter('h_factor', val=38.0, units=None, opt=False, dynamic=False,
                          desc='corrective factor for convective heat transfer coefficients')

# Atmosphere
traj.add_design_parameter('T0', val=35.0+273.15, opt=False, dynamic=False,
                          desc='ambient temperature', units='K')
traj.add_design_parameter('g', val=9.81, units='m/s**2', opt=False, dynamic=False)



# Set the discretization options
nun_seg=5

# First phase: HLP run-up
transcription = dm.Radau(num_segments=num_seg, order=5, segment_ends=seg_ends, compressed=False)
phase1 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p1 = traj.add_phase('phase1', phase1)
traj_p1.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[0],
                         duration_adder=dt_seg[0])
traj_p1.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[0], v_seg[1], 5),
                    targets=['u_inf'])
traj_p1.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[0], h_seg[1], 5),
                    targets=['h'])
traj_p1.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[0], p_seg[1], 5),
                    targets=['throttle'])

traj_p1.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=True, fix_final=False)
traj_p1.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=True, fix_final=False)
traj_p1.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=True, fix_final=False)
traj_p1.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=True, fix_final=False)

# Second phase: go/no-go
phase2 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p2 = traj.add_phase('phase2', phase2)
traj_p2.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[1],
                         duration_adder=dt_seg[1])
traj_p2.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[1], v_seg[2], 5))
traj_p2.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[1], h_seg[2], 5))
traj_p2.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[1], p_seg[2], 5))

traj_p2.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p2.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p2.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p2.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Third phase: ground roll
phase3 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p3 = traj.add_phase('phase3', phase3)
traj_p3.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[2],
                         duration_adder=dt_seg[2])
traj_p3.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[2], v_seg[3], 5))
traj_p3.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[2], h_seg[3], 5))
traj_p3.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[2], p_seg[3], 5))

traj_p3.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p3.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p3.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p3.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Fourth phase: climb to 1500'
phase4 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p4 = traj.add_phase('phase4', phase4)
traj_p4.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[3],
                         duration_adder=dt_seg[3])
traj_p4.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[3], v_seg[4], 5))
traj_p4.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[3], h_seg[4], 5))
traj_p4.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[3], p_seg[4], 5))

traj_p4.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p4.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p4.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p4.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Fifth phase: cruise climb
phase5 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p5 = traj.add_phase('phase5', phase5)
traj_p5.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[4],
                         duration_adder=dt_seg[4])
traj_p5.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[4], v_seg[5], 5))
traj_p5.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[4], h_seg[5], 5))
traj_p5.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[4], p_seg[5], 5))

traj_p5.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p5.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p5.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p5.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Sixth phase: cruise
phase6 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p6 = traj.add_phase('phase6', phase6)
traj_p6.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[5],
                         duration_adder=dt_seg[5])
traj_p6.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[5], v_seg[6], 5))
traj_p6.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[5], h_seg[6], 5))
traj_p6.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[5], p_seg[6], 5))

traj_p6.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p6.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p6.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p6.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Seventh phase: descent to 1500'
phase7 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p7 = traj.add_phase('phase7', phase7)
traj_p7.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[6],
                         duration_adder=t_seg[6])
traj_p7.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[6], v_seg[7], 5))
traj_p7.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[6], h_seg[7], 5))
traj_p7.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[6], p_seg[7], 5))

traj_p7.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p7.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p7.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p7.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Eighth phase: final approach
phase8 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p8 = traj.add_phase('phase8', phase8)
traj_p8.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[7],
                         duration_adder=dt_seg[7])
traj_p8.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[7], v_seg[8], 5))
traj_p8.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[7], h_seg[8], 5))
traj_p8.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[7], p_seg[8], 5))

traj_p8.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p8.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p8.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p8.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Ninth phase: go around to 1500'
phase9 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p9 = traj.add_phase('phase9', phase9)
traj_p9.set_time_options(fix_initial=True, fix_duration=True,
                         initial_adder=t_seg[8],
                         duration_adder=dt_seg[8])
traj_p9.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[8], v_seg[9], 5))
traj_p9.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[8], h_seg[9], 5))
traj_p9.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[8], p_seg[9], 5))

traj_p9.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p9.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p9.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p9.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Tenth phase: approach pattern
phase10 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p10 = traj.add_phase('phase10', phase10)
traj_p10.set_time_options(fix_initial=True, fix_duration=True,
                          initial_adder=t_seg[9],
                          duration_adder=dt_seg[9])
traj_p10.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[9], v_seg[10], 5))
traj_p10.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[9], h_seg[10], 5))
traj_p10.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[9], p_seg[10], 5))

traj_p10.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p10.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p10.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p10.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Eleventh phase: final approach
phase11 = dm.Phase(ode_class=MotorTempODE, transcription=transcription)
traj_p11 = traj.add_phase('phase11', phase11)
traj_p11.set_time_options(fix_initial=True, fix_duration=True,
                          initial_adder=t_seg[10],
                          duration_adder=dt_seg[10])
traj_p11.add_control(name='velocity', units='m/s',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(v_seg[10], v_seg[11], 5))
traj_p11.add_control(name='altitude', units='m',
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(h_seg[10], h_seg[11], 5))
traj_p11.add_control(name='throttle', units=None,
                    fix_initial=True, fix_final=True,
                    opt=False,
                    val=np.linspace(p_seg[10], p_seg[11], 5))

traj_p11.add_state('T_m', targets=['T_m'],
                  rate_source='Tdot_m',
                  units='K', fix_initial=False, fix_final=False)
traj_p11.add_state('T_s', targets=['T_s'],
                  rate_source='Tdot_s',
                  units='K', fix_initial=False, fix_final=False)
traj_p11.add_state('T_p', targets=['T_p'],
                  rate_source='Tdot_p',
                  units='K', fix_initial=False, fix_final=False)
traj_p11.add_state('T_a', targets=['T_a'],
                  rate_source='Tdot_a',
                  units='K', fix_initial=False, fix_final=False)

# Link all of the phases
link_vars = ['time', 'velocity', 'altitude',
             'T_m', 'T_s', 'T_p', 'T_a']
             # 'Qdot_in', 'Qdot_s', 'Qdot_p', 'Qdot_c', 'Qdot_cs',
             # 'Tdot_m', 'Tdot_s', 'Tdot_p', 'Tdot_a'
traj.link_phases(phases=['phase1', 'phase2'], vars=link_vars)
traj.link_phases(phases=['phase2', 'phase3'], vars=link_vars)
traj.link_phases(phases=['phase3', 'phase4'], vars=link_vars)
traj.link_phases(phases=['phase4', 'phase5'], vars=link_vars)
traj.link_phases(phases=['phase5', 'phase6'], vars=link_vars)
traj.link_phases(phases=['phase6', 'phase7'], vars=link_vars)
traj.link_phases(phases=['phase7', 'phase8'], vars=link_vars)
traj.link_phases(phases=['phase8', 'phase9'], vars=link_vars)
traj.link_phases(phases=['phase9', 'phase10'], vars=link_vars)
traj.link_phases(phases=['phase10', 'phase11'], vars=link_vars)

prob.model.options['assembled_jac_type'] = 'csc'
prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

# Run the problem
prob.setup()
prob.run_driver()

