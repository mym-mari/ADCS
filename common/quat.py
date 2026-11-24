# common/quat.py
"""Algebra de cuaterniones. Convencion: q = [x, y, z, w], norma 1.
q_BI rota vectores de ECI a B:  v_b = dcm(q) @ v_eci."""
import numpy as np


def normalize(q):
    return q / np.linalg.norm(q)


def conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def mult(p, q):
    """Producto p (x) q: aplicar q y luego p (composicion de rotaciones)."""
    pv, pw = p[:3], p[3]
    qv, qw = q[:3], q[3]
    v = pw * qv + qw * pv + np.cross(pv, qv)
    w = pw * qw - pv @ qv
    return np.array([v[0], v[1], v[2], w])


def skew(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def dcm(q):
    """Matriz de rotacion ECI -> B."""
    v, w = q[:3], q[3]
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) - 2.0 * w * skew(v)


def from_rotvec(rv):
    """Cuaternion de un vector de rotacion (rad)."""
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return normalize(np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0]))
    ax = rv / ang
    s = np.sin(0.5 * ang)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(0.5 * ang)])


def omega_matrix(w):
    """Omega(w) tal que q_dot = 0.5 * Omega(w) @ q."""
    return np.block([[-skew(w), w.reshape(3, 1)],
                     [-w.reshape(1, 3), np.zeros((1, 1))]])


def error_quat(q_ref, q):
    """q_e = q (x) q_ref^-1: rotacion de la referencia al cuerpo."""
    qe = mult(q, conj(q_ref))
    return qe if qe[3] >= 0 else -qe   # camino corto
