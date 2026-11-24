# plant/environment_plant.py
"""Planta de verdad (Mariana). Orbita con J2, cuerpo rigido con ruedas,
gradiente gravitacional, campo magnetico, Sol/eclipse y sensores.
Implementa PlantAPI de interfaces.py. Etapa 3 (arrastre): ver disturbance_torque()."""
import numpy as np

import params as P
from common import quat as Q
from interfaces import TruthState, SensorPacket, ActuatorCmd

# Campo magnetico: dipolo inclinado, fijo a la Tierra (rota con OMEGA_E)
B0 = 3.12e-5                               # T, ecuador en la superficie
_TH = np.deg2rad(9.3)                      # colatitud del polo geomagnetico norte
_LA = np.deg2rad(-72.7)                    # longitud del polo geomagnetico norte
M_HAT_ECEF = -np.array([np.sin(_TH) * np.cos(_LA),
                        np.sin(_TH) * np.sin(_LA),
                        np.cos(_TH)])      # el momento dipolar apunta al polo sur

# Sol: orbita circular aparente de la Tierra, equinoccio en t = 0
EPS = np.deg2rad(23.44)
N_SUN = 2.0 * np.pi / (365.25 * 86400.0)   # rad/s


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def sun_eci(t):
    lam = N_SUN * t
    return np.array([np.cos(lam), np.sin(lam) * np.cos(EPS), np.sin(lam) * np.sin(EPS)])


def b_eci(t, r):
    """Campo magnetico (T) en ECI, dipolo inclinado."""
    rn = np.linalg.norm(r)
    rh = r / rn
    m = rot_z(P.OMEGA_E * t) @ M_HAT_ECEF
    return B0 * (P.R_E / rn) ** 3 * (3.0 * (m @ rh) * rh - m)


def in_eclipse(r, s):
    """Sombra cilindrica."""
    proj = r @ s
    return bool(proj < 0.0 and np.linalg.norm(r - proj * s) < P.R_E)


class Plant:
    def __init__(self):
        self.enable_gg = True              # gradiente gravitacional (apagable para pruebas)
        self._Jinv = np.linalg.inv(P.J_B)
        self.reset(P.SEED)

    # ------------------------------------------------------------------ API
    def reset(self, seed=P.SEED, tumble=True):
        self.rng = np.random.default_rng(seed)
        R = rot_z(P.RAAN0) @ rot_x(P.INC)
        vc = np.sqrt(P.MU / P.A_ORB)
        r = R @ np.array([P.A_ORB, 0., 0.])
        v = R @ np.array([0., vc, 0.])
        q = self.rng.normal(size=4)
        q = Q.normalize(q)
        if q[3] < 0:
            q = -q
        w = self.rng.uniform(-P.OMEGA0_MAX, P.OMEGA0_MAX, 3) if tumble else np.zeros(3)
        self.x = np.concatenate([r, v, q, w, np.zeros(3)])   # [r, v, q, omega, h_rw]
        self.t = 0.0
        self.cmd = ActuatorCmd()
        self.gyro_bias = self.rng.normal(0.0, P.GYRO_BIAS0_STD, 3)
        return self.truth()

    def step(self, cmd, dt=P.DT_PLANT):
        """Avanza dt con la orden cmd (saturada aqui). Devuelve la verdad."""
        self.cmd = self._saturate(cmd)
        x, t = self.x, self.t
        k1 = self._deriv(t, x)
        k2 = self._deriv(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = self._deriv(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = self._deriv(t + dt, x + dt * k3)
        self.x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x[6:10] = Q.normalize(self.x[6:10])
        self.t = t + dt
        return self.truth()

    def truth(self):
        x = self.x
        return TruthState(t=self.t, q=x[6:10].copy(), omega=x[10:13].copy(),
                          h_rw=x[13:16].copy(), r_eci=x[0:3].copy(), v_eci=x[3:6].copy(),
                          in_eclipse=in_eclipse(x[0:3], sun_eci(self.t)))

    def sense(self):
        """Paquete de sensores (llamar cada DT_FSW). Ruido, sesgo y cuantizacion."""
        x, rng = self.x, self.rng
        A = Q.dcm(x[6:10])
        ecl = in_eclipse(x[0:3], sun_eci(self.t))

        mag = A @ b_eci(self.t, x[0:3]) + rng.normal(0.0, P.MAG_NOISE_STD, 3)
        mag = np.round(mag / P.MAG_QUANT) * P.MAG_QUANT

        if ecl:
            sun = np.full(3, np.nan)
        else:
            sun = A @ sun_eci(self.t) + rng.normal(0.0, P.SUN_NOISE_STD, 3)
            sun = sun / np.linalg.norm(sun)

        self.gyro_bias = self.gyro_bias + P.GYRO_RRW * np.sqrt(P.DT_FSW) * rng.normal(size=3)
        gyro = x[10:13] + self.gyro_bias + P.GYRO_ARW / np.sqrt(P.DT_FSW) * rng.normal(size=3)

        return SensorPacket(t=self.t, mag_b=mag, sun_b=sun, gyro_b=gyro,
                            eclipse_flag=ecl, h_rw_meas=x[13:16].copy())

    # ------------------------------------------------------------ internals
    def _saturate(self, cmd):
        tau = np.clip(cmd.tau_rw, -P.TAU_RW_MAX, P.TAU_RW_MAX)
        h = self.x[13:16]
        # h_dot = -tau: si la rueda esta al limite y la orden la empuja mas, se anula
        for i in range(3):
            if abs(h[i]) >= P.H_RW_MAX and np.sign(-tau[i]) == np.sign(h[i]):
                tau[i] = 0.0
        m = np.clip(cmd.m_mtq, -P.M_MTQ_MAX, P.M_MTQ_MAX)
        return ActuatorCmd(t=cmd.t, tau_rw=tau, m_mtq=m)

    def disturbance_torque(self, t, x):
        """Torque de arrastre en B (N m). Etapa 3: aqui va el modelo por facetas."""
        return np.zeros(3)

    def _deriv(self, t, x):
        r, v, q, w, h = x[0:3], x[3:6], x[6:10], x[10:13], x[13:16]
        rn = np.linalg.norm(r)

        # Traslacion: gravedad central + J2
        f = 1.5 * P.J2 * P.MU * P.R_E ** 2 / rn ** 5
        z2 = (r[2] / rn) ** 2
        a = -P.MU * r / rn ** 3 + f * np.array([-r[0] * (1 - 5 * z2),
                                                -r[1] * (1 - 5 * z2),
                                                -r[2] * (3 - 5 * z2)])

        # Actitud
        A = Q.dcm(q)
        qdot = 0.5 * Q.omega_matrix(w) @ q
        tau_mtq = np.cross(self.cmd.m_mtq, A @ b_eci(t, r))
        tau_gg = np.zeros(3)
        if self.enable_gg:
            rb = A @ (r / rn)
            tau_gg = 3.0 * P.MU / rn ** 3 * np.cross(rb, P.J_B @ rb)
        tau_ext = tau_mtq + tau_gg + self.disturbance_torque(t, x)
        wdot = self._Jinv @ (self.cmd.tau_rw + tau_ext - np.cross(w, P.J_B @ w + h))
        hdot = -self.cmd.tau_rw

        return np.concatenate([v, a, qdot, wdot, hdot])


# ---------------------------------------------------------------- verificacion
if __name__ == "__main__":
    # Ejecutar desde la raiz del repo:  python -m plant.environment_plant
    pl = Plant()
    zero = ActuatorCmd()
    T = 2 * np.pi * np.sqrt(P.A_ORB ** 3 / P.MU)
    print(f"Periodo kepleriano: {T / 60:.2f} min (esperado ~94.6)")

    # 1) Una orbita con paso de 1 s: radio, campo magnetico y eclipse
    pl.reset(tumble=True)
    r0 = pl.x[0:3].copy()
    rs, bs, ecl = [], [], 0
    n = int(T)
    for _ in range(n):
        s = pl.step(zero, dt=1.0)
        rs.append(np.linalg.norm(s.r_eci))
        bs.append(np.linalg.norm(b_eci(s.t, s.r_eci)))
        ecl += s.in_eclipse
    ang = np.degrees(np.arccos(r0 @ pl.x[0:3] / (np.linalg.norm(r0) * np.linalg.norm(pl.x[0:3]))))
    print(f"Radio: min {min(rs) / 1e3:.1f} km, max {max(rs) / 1e3:.1f} km (esperado ~6878)")
    print(f"|B|: min {min(bs) * 1e6:.1f} uT, max {max(bs) * 1e6:.1f} uT (esperado ~25 a 50)")
    print(f"Fraccion en eclipse: {100 * ecl / n:.1f} % (esperado ~35 % con el Sol en el plano orbital)")
    print(f"Angulo de regreso tras 1 periodo: {ang:.2f} deg (pequeno)")

    # 2) Conservacion del momento angular total (sin gradiente, sin magnetorquers)
    pl.reset(tumble=True)
    pl.enable_gg = False
    H0 = Q.dcm(pl.x[6:10]).T @ (P.J_B @ pl.x[10:13] + pl.x[13:16])
    for _ in range(600):
        pl.step(zero, dt=0.1)
    H1 = Q.dcm(pl.x[6:10]).T @ (P.J_B @ pl.x[10:13] + pl.x[13:16])
    print(f"Deriva de momento angular en 60 s: {np.linalg.norm(H1 - H0) / np.linalg.norm(H0):.2e} (esperado < 1e-6)")

    # 3) Sensores
    pl.reset()
    pkt = pl.sense()
    print("mag_b (T):", pkt.mag_b, "| sun_b:", pkt.sun_b, "| gyro_b:", pkt.gyro_b)
