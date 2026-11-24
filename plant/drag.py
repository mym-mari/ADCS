# plant/drag.py
"""Arrastre aerodinamico por facetas para el 6U (Mariana). Etapa 3, parte 1.
Uso:  from plant.drag import DragPlant   (en lugar de Plant)
Verificar desde la raiz:  python -m plant.drag"""
import numpy as np

import params as P
from common import quat as Q
from plant.environment_plant import Plant

# Geometria del 6U (decisiones del equipo, ajustables)
L = np.array([0.100, 0.226, 0.366])        # m, dimensiones en x, y, z del cuerpo
R_CM = np.array([0.003, 0.002, 0.005])     # m, centro de masa respecto al centro geometrico

# Atmosfera exponencial
RHO0 = 6.0e-13                             # kg/m^3 a H_REF
H_REF = 500e3                              # m
H_SCALE = 63e3                             # m

# Seis caras: +x, +y, +z, -x, -y, -z
NORMALS = np.vstack([np.eye(3), -np.eye(3)])
AREAS = np.array([L[1] * L[2], L[0] * L[2], L[0] * L[1]] * 2)
CENTERS = NORMALS * (L / 2.0)              # centro de cada cara, desde el centro geometrico


def density(r_norm):
    """Densidad atmosferica (kg/m^3) a la distancia r_norm del centro de la Tierra."""
    return RHO0 * np.exp(-((r_norm - P.R_E) - H_REF) / H_SCALE)


def projected_area(vhat_b):
    """Area proyectada (m^2) normal a la velocidad relativa, dada en el cuerpo."""
    vh = np.asarray(vhat_b, dtype=float)
    vh = vh / np.linalg.norm(vh)
    return float(np.sum(AREAS * np.maximum(0.0, NORMALS @ vh)))


def drag_force_torque(vrel_b, rho):
    """Fuerza (N) y torque (N m) de arrastre en el cuerpo. vrel_b: velocidad relativa al aire, en B."""
    v = np.linalg.norm(vrel_b)
    vh = vrel_b / v
    proj = np.maximum(0.0, NORMALS @ vh)
    F_i = -0.5 * rho * P.CD * v ** 2 * (AREAS * proj)[:, None] * vh      # (6, 3)
    F = F_i.sum(axis=0)
    tau = np.cross(CENTERS - R_CM, F_i).sum(axis=0)
    return F, tau


def decay_rate(rho, area, a=P.A_ORB):
    """Decaimiento de semieje mayor (m/s) en orbita circular."""
    return rho * P.CD * area / P.MASS * np.sqrt(P.MU * a)


class DragPlant(Plant):
    """Plant con torque de arrastre aerodinamico."""

    def disturbance_torque(self, t, x):
        r, v, q = x[0:3], x[3:6], x[6:10]
        vrel = v - np.cross(np.array([0.0, 0.0, P.OMEGA_E]), r)
        vb = Q.dcm(q) @ vrel
        _, tau = drag_force_torque(vb, density(np.linalg.norm(r)))
        return tau


# ---------------------------------------------------------------- verificacion
if __name__ == "__main__":
    from interfaces import ActuatorCmd

    rn = P.A_ORB
    rho = density(rn)
    vc = np.sqrt(P.MU / rn)
    print(f"Densidad a 500 km: {rho:.2e} kg/m^3 (esperado 6.00e-13)")
    print(f"Areas de cara: x {AREAS[0]:.4f}, y {AREAS[1]:.4f}, z {AREAS[2]:.4f} m^2")
    print()
    for name, vh in [("velocidad por +x (nadir)", [1., 0., 0.]),
                     ("velocidad por +y", [0., 1., 0.]),
                     ("velocidad por +z (eje largo, minimo)", [0., 0., 1.])]:
        vh = np.array(vh)
        A = projected_area(vh)
        F, tau = drag_force_torque(vc * vh, rho)
        print(f"{name}: A = {A:.4f} m^2, |F| = {np.linalg.norm(F):.2e} N, "
              f"|tau| = {np.linalg.norm(tau):.2e} N m, "
              f"decaimiento = {decay_rate(rho, A) * 86400:.1f} m/dia")
    print("(esperado: A = 0.0827, 0.0366, 0.0226; decaimiento ~62, ~27 y ~17 m/dia)")

    # La planta con arrastre corre y produce torque con la actitud real
    pl = DragPlant()
    pl.reset(tumble=False)
    for _ in range(1000):
        pl.step(ActuatorCmd(), dt=0.01)
    tau = pl.disturbance_torque(pl.t, pl.x)
    print(f"\nDragPlant, torque de arrastre tras 10 s: {tau} N m")
