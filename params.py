# params.py
"""Parametros unicos del proyecto FACSAT-LEO. SI, radianes, tesla.
Los valores marcados TODO son provisionales: acordarlos entre los tres."""
import numpy as np

# Constantes
MU = 3.986004418e14        # m^3/s^2
R_E = 6378.137e3           # m
J2 = 1.08262668e-3
OMEGA_E = 7.2921159e-5     # rad/s

# Orbita SSO circular
ALT = 500e3                # m
INC = np.deg2rad(97.4)     # rad
A_ORB = R_E + ALT          # m

# Satelite 6U (propuesta)
MASS = 8.0                 # kg
J_B = np.diag([0.04, 0.04, 0.015])   # kg m^2
A_FACE = 0.0226            # m^2, cara corta alineada con la velocidad
CD = 2.2

# Tiempos
DT_PLANT = 0.01            # s (100 Hz)
DT_FSW = 0.1               # s (10 Hz)

# Actuadores (TODO)
H_RW_MAX = 0.01            # N m s por rueda
TAU_RW_MAX = 1e-3          # N m por rueda
M_MTQ_MAX = 0.2            # A m^2 por eje
DUMP_FRACTION = 0.6        # umbral de descarga de momentum

# Sensores (TODO, los fija quien escribe los modelos de sensores)
MAG_NOISE_STD = 1e-7       # T
MAG_QUANT = 1e-8           # T
SUN_NOISE_STD = np.deg2rad(0.1)      # rad
GYRO_ARW = 1e-4            # rad/s/sqrt(Hz)
GYRO_RRW = 1e-6            # rad/s^2/sqrt(Hz)
GYRO_BIAS0_STD = np.deg2rad(0.05)    # rad/s

# Requisitos
DETUMBLE_RATE_MAX = np.deg2rad(0.5)  # rad/s
POINTING_ERR_MAX = np.deg2rad(1.0)   # rad (3 sigma)
EST_ERR_SUN = np.deg2rad(0.5)        # rad
EST_ERR_ECLIPSE = np.deg2rad(3.0)    # rad

# Reproducibilidad y condiciones iniciales
SEED = 42
T_SIM = 10 * 94.6 * 60.0                 # s, 10 orbitas
RAAN0 = 0.0                              # rad (TODO: fijar con la LTAN deseada)
EPOCH_JD = 2461000.5                     # fecha juliana de referencia (TODO)
OMEGA0_MAX = np.deg2rad(5.0)             # rad/s por eje, tumbling inicial
