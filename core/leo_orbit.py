# ======================================
# core/leo_orbit.py
# Utilities for reading Starlink TLEs and computing ECI/ECEF/ENU states
# ======================================

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional

try:
    from sgp4.api import Satrec, jday
except Exception as e:
    raise ImportError("Please install sgp4: pip install sgp4") from e

# ---- Earth constants ----
W_EARTH = 7.2921150e-5        # rad/s (Earth rotation)
R_EARTH = 6378137.0           # m
C = 299_792_458.0

@dataclass
class TLE:
    name: str
    line1: str
    line2: str

@dataclass
class SatState:
    name: str
    r_eci_m: np.ndarray   # (3,)
    v_eci_mps: np.ndarray # (3,)
    r_ecef_m: np.ndarray  # (3,)
    v_ecef_mps: np.ndarray# (3,)

# ---------------------------
# TLE I/O
# ---------------------------
def read_tle_file(path: str) -> List[TLE]:
    """Read TLE file in triplets: name, L1, L2."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        l1 = lines[i+1]
        l2 = lines[i+2]
        # Basic sanity
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            # If file is name-less (2-line-only), fabricate names
            name = f"SAT_{i//3:05d}"
            l1 = lines[i]
            l2 = lines[i+1]
            i += 2
        else:
            i += 3
        out.append(TLE(name=name, line1=l1, line2=l2))
    return out

# ---------------------------
# Time utils
# ---------------------------
def to_jday(dt: datetime) -> Tuple[int, float]:
    dt = dt.astimezone(timezone.utc)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
    return jd, fr

# ---------------------------
# Frames: ECI -> ECEF
# ---------------------------
def _gmst(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time (rad), low-order for real-time ops."""
    # Simplified GMST (sufficient for RF Doppler/visibility routing)
    # Source: Vallado (approx)
    dt = dt.astimezone(timezone.utc)
    JD = (dt - datetime(2000,1,1,tzinfo=timezone.utc)).total_seconds()/86400.0 + 2451544.5
    T  = (JD - 2451545.0)/36525.0
    GMST_sec = 67310.54841 + (876600.0*3600 + 8640184.812866)*T + 0.093104*(T**2) - 6.2e-6*(T**3)
    GMST_rad = np.deg2rad((GMST_sec/240.0) % 360.0)
    return GMST_rad

def eci_to_ecef(r_eci: np.ndarray, v_eci: np.ndarray, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate ECI to ECEF (simple: no polar motion; includes Earth rotation for velocity)."""
    theta = _gmst(dt)
    Rz = np.array([[ np.cos(theta),  np.sin(theta), 0.0],
                   [-np.sin(theta),  np.cos(theta), 0.0],
                   [ 0.0,            0.0,          1.0]])
    r_ecef = Rz @ r_eci
    omega = np.array([0, 0, W_EARTH])
    v_ecef = Rz @ (v_eci - np.cross(omega, r_eci))
    return r_ecef, v_ecef

# ---------------------------
# Propagation via SGP4
# ---------------------------
def propagate_tle(tle: TLE, dt: datetime) -> SatState:
    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    jd, fr = to_jday(dt)
    e, r_km, v_kmps = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error code: {e}")
    r_eci = np.array(r_km,   dtype=float) * 1e3
    v_eci = np.array(v_kmps, dtype=float) * 1e3
    r_ecef, v_ecef = eci_to_ecef(r_eci, v_eci, dt)
    return SatState(name=tle.name, r_eci_m=r_eci, v_eci_mps=v_eci, r_ecef_m=r_ecef, v_ecef_mps=v_ecef)

# ---------------------------
# ENU utilities (optional)
# ---------------------------
def ecef_to_enu(r_ecef: np.ndarray, ref_llh_deg: Tuple[float,float,float]) -> np.ndarray:
    """EC(E)F -> ENU vector (m) relative to ref_llh (lat,lon,h[m])."""
    lat, lon, h = np.deg2rad(ref_llh_deg[0]), np.deg2rad(ref_llh_deg[1]), ref_llh_deg[2]
    # Approx ECEF of ref (simple spherical Earth for routing)
    cl, sl = np.cos(lat), np.sin(lat)
    ce, se = np.cos(lon), np.sin(lon)
    R = R_EARTH + h
    r0 = np.array([R*cl*ce, R*cl*se, R*sl])
    dr = r_ecef - r0
    T = np.array([[-se,        ce,       0],
                  [-ce*sl, -se*sl,   cl],
                  [ ce*cl,  se*cl,   sl]])
    enu = T @ dr
    return enu

def radial_velocity_hz(tx_pos_ecef: np.ndarray, rx_pos_ecef: np.ndarray,
                       tx_vel_ecef: np.ndarray, rx_vel_ecef: np.ndarray,
                       f_c: float) -> float:
    """Doppler (Hz) from relative radial velocity along line-of-sight."""
    los = rx_pos_ecef - tx_pos_ecef
    u = los / (np.linalg.norm(los) + 1e-12)
    v_rel = (rx_vel_ecef - tx_vel_ecef)
    fd = (np.dot(v_rel, u) / C) * f_c
    return float(fd)
