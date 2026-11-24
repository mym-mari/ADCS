# interfaces.py
"""Contrato entre planta, software de vuelo (FSW) y control.

CONVENCIONES (validas en todo el repo)
- Marcos: ECI (inercial), LVLH (orbital), B (cuerpo).
- Cuaternion q = [x, y, z, w] (escalar al final), norma 1, rota de ECI a B (q_BI).
- omega: velocidad angular de B respecto a ECI, expresada en B, rad/s.
- Vectores de sensores y torques: expresados en B. Unidades SI.
- LVLH: z hacia el nadir, y opuesto al momento angular orbital, x completa la terna derecha.
- tau_rw: par que las ruedas ejercen SOBRE EL CUERPO (N m). Dinamica: h_rw_dot = -tau_rw.
- m_mtq: dipolo de los magnetorquers (A m^2); par sobre el cuerpo = m x B.
- La planta es la unica que ve TruthState. El FSW solo recibe SensorPacket.
- Tasas: planta 100 Hz, FSW 10 Hz. Las ordenes se mantienen constantes entre llamadas del FSW.
"""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


def _v3():
    return np.zeros(3)


def _q_id():
    return np.array([0., 0., 0., 1.])


@dataclass
class TruthState:
    """Salida de la planta (100 Hz). Prohibido usarla en el FSW."""
    t: float = 0.0
    q: np.ndarray = field(default_factory=_q_id)
    omega: np.ndarray = field(default_factory=_v3)    # rad/s, en B
    h_rw: np.ndarray = field(default_factory=_v3)     # N m s, en B
    r_eci: np.ndarray = field(default_factory=_v3)    # m
    v_eci: np.ndarray = field(default_factory=_v3)    # m/s
    in_eclipse: bool = False


@dataclass
class SensorPacket:
    """Planta -> FSW (10 Hz)."""
    t: float = 0.0
    mag_b: np.ndarray = field(default_factory=_v3)    # T, en B
    sun_b: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))  # unitario, NaN en eclipse
    gyro_b: np.ndarray = field(default_factory=_v3)   # rad/s, en B
    eclipse_flag: bool = False
    h_rw_meas: np.ndarray = field(default_factory=_v3)  # N m s, en B


@dataclass
class EstimatorOutput:
    """Estimador -> control y gestor de modos (10 Hz)."""
    t: float = 0.0
    q_hat: np.ndarray = field(default_factory=_q_id)
    omega_hat: np.ndarray = field(default_factory=_v3)  # gyro menos sesgo, rad/s
    bias_hat: np.ndarray = field(default_factory=_v3)   # rad/s
    P: np.ndarray = field(default_factory=lambda: np.eye(6))  # [d_theta(3), d_bias(3)]
    fallback_safe: bool = False   # True si la covarianza de actitud supera el umbral


@dataclass
class ActuatorCmd:
    """FSW -> planta (10 Hz)."""
    t: float = 0.0
    tau_rw: np.ndarray = field(default_factory=_v3)   # N m
    m_mtq: np.ndarray = field(default_factory=_v3)    # A m^2


class Mode:
    DETUMBLE = "detumble"
    IMAGING = "imaging"
    IDLE = "idle"
    SAFE = "safe"


class PlantAPI(Protocol):
    """Lo que implementa plant/environment_plant.py (Mariana)."""
    def reset(self, seed: int) -> TruthState: ...
    def step(self, cmd: ActuatorCmd) -> TruthState:
        """Avanza DT_PLANT con la orden vigente."""
    def sense(self) -> SensorPacket:
        """Genera el paquete de sensores con ruido, sesgo y cuantizacion."""


class FlightSoftwareAPI(Protocol):
    """Lo que implementa fsw/flight_software.py (Diego, con el estimador de Ricardo)."""
    def reset(self) -> None: ...
    def step(self, pkt: SensorPacket) -> tuple[ActuatorCmd, EstimatorOutput, str]:
        """Devuelve la orden, la salida del estimador y el modo vigente."""
