# plant/test_plant.py
"""Pruebas de la planta. Ejecutar desde la raiz del repo:  python -m plant.test_plant
Tarda uno o dos minutos (la precesion simula un dia completo)."""
import numpy as np

import params as P
from common import quat as Q
from interfaces import ActuatorCmd
from plant.environment_plant import Plant, sun_eci, b_eci, MAG_BIAS_STD

zero = ActuatorCmd()


def check(name, value, expected, tol_rel, unit=""):
    ok = abs(value - expected) <= tol_rel * abs(expected)
    print(f"[{'OK' if ok else 'FALLA'}] {name}: {value:.6g} {unit} (esperado {expected:.6g}, tol {100 * tol_rel:.0f} %)")
    return ok


results = []

# ---------------------------------------------------------------- 1) Precesion del nodo
print("1) Precesion del nodo (1 dia, paso 2 s)...")
pl = Plant()
pl.reset(tumble=False)
raan, tt = [], []
for k in range(43200):
    pl.step(zero, dt=2.0)
    if k % 30 == 0:
        h = np.cross(pl.x[0:3], pl.x[3:6])
        raan.append(np.arctan2(h[0], -h[1]))
        tt.append(pl.t)
raan = np.unwrap(np.array(raan))
slope = np.polyfit(np.array(tt) / 86400.0, np.degrees(raan), 1)[0]      # deg/dia
n_mm = np.sqrt(P.MU / P.A_ORB ** 3)
teo = np.degrees(-1.5 * P.J2 * n_mm * (P.R_E / P.A_ORB) ** 2 * np.cos(P.INC)) * 86400.0
results.append(check("Precesion del nodo", slope, teo, 0.03, "deg/dia"))
print(f"     (referencia SSO: 0.9856 deg/dia)")

# ---------------------------------------------------------------- 2) Energia cinetica
print("2) Conservacion de energia (sin torques, 60 s)...")
pl = Plant()
pl.reset(tumble=True)
pl.enable_gg = False
T0 = 0.5 * pl.x[10:13] @ (P.J_B @ pl.x[10:13])
for _ in range(600):
    pl.step(zero, dt=0.1)
T1 = 0.5 * pl.x[10:13] @ (P.J_B @ pl.x[10:13])
drift = abs(T1 - T0) / T0
print(f"[{'OK' if drift < 1e-6 else 'FALLA'}] Deriva relativa de energia: {drift:.2e} (esperado < 1e-6)")
results.append(drift < 1e-6)
qn = np.linalg.norm(pl.x[6:10])
print(f"[{'OK' if abs(qn - 1) < 1e-12 else 'FALLA'}] Norma del cuaternion: {qn:.15f}")
results.append(abs(qn - 1) < 1e-12)

# ---------------------------------------------------------------- 3) Estadisticas de sensores
print("3) Estadisticas de sensores (1 orbita, un paquete por segundo)...")
pl = Plant()
pl.reset(seed=123, tumble=True)
Torb = 2 * np.pi * np.sqrt(P.A_ORB ** 3 / P.MU)
mag_res, sun_ang, gyro_res, bias_inc, nan_ok = [], [], [], [], True
prev_bias = pl.gyro_bias.copy()
for _ in range(int(Torb)):
    pl.step(zero, dt=1.0)
    pkt = pl.sense()
    A = Q.dcm(pl.x[6:10])
    mag_res.append(pkt.mag_b - A @ b_eci(pl.t, pl.x[0:3]))
    gyro_res.append(pkt.gyro_b - pl.x[10:13] - pl.gyro_bias)
    bias_inc.append(pl.gyro_bias - prev_bias)
    prev_bias = pl.gyro_bias.copy()
    if pkt.eclipse_flag:
        nan_ok &= bool(np.all(np.isnan(pkt.sun_b)))
    else:
        nan_ok &= not np.any(np.isnan(pkt.sun_b))
        s_true = A @ sun_eci(pl.t)
        sun_ang.append(np.arccos(np.clip(pkt.sun_b @ s_true, -1.0, 1.0)))
mag_res, gyro_res, bias_inc = np.array(mag_res), np.array(gyro_res), np.array(bias_inc)

print(f"[{'OK' if nan_ok else 'FALLA'}] Sol: NaN si y solo si hay eclipse")
results.append(nan_ok)

bias_err = np.max(np.abs(mag_res.mean(axis=0) - pl.mag_bias))
print(f"[{'OK' if bias_err < 1e-8 else 'FALLA'}] Magnetometro, sesgo estimado vs aplicado: error max {bias_err:.2e} T "
      f"(sesgo aplicado {np.linalg.norm(pl.mag_bias):.2e} T, sigma de sesgo {MAG_BIAS_STD:.1e} T)")
results.append(bias_err < 1e-8)

# El ruido se mide DESPUES de quitar la media de cada eje (el sesgo es distinto por eje)
mag_std_teo = np.sqrt(P.MAG_NOISE_STD ** 2 + P.MAG_QUANT ** 2 / 12.0)
mag_noise = (mag_res - mag_res.mean(axis=0)).std()
results.append(check("Magnetometro, desviacion del ruido", mag_noise, mag_std_teo, 0.05, "T"))

results.append(check("Sensor solar, error angular rms", np.sqrt(np.mean(np.square(sun_ang))),
                     np.sqrt(2.0) * P.SUN_NOISE_STD, 0.10, "rad"))

results.append(check("Giroscopo, desviacion del ruido", gyro_res.std(), P.GYRO_ARW / np.sqrt(P.DT_FSW), 0.05, "rad/s"))

results.append(check("Giroscopo, incremento del sesgo por paso", bias_inc.std(),
                     P.GYRO_RRW * np.sqrt(P.DT_FSW), 0.10, "rad/s"))

print()
print("TODAS LAS PRUEBAS OK" if all(results) else "HAY PRUEBAS QUE FALLAN: pegame la salida completa")
