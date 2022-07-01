import numpy as np
from decimal import Decimal

TIME_UNITS = "milliseconds"
ANGLE_UNITS = "degrees"
DISTANCE_UNITS = "meters"

time_scale_factors = {
    "seconds": 1.0, # everything defined w.r.t. seconds
    "milliseconds": 1.0e3,
    "miliseconds": 1.0e3, # accounting for soundata typo
    "nanoseconds": 1.0e9,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
    "years": 31536000.0, # not accounting for skip years :p
}

angle_scale_factors = {
    "degrees": 1.0, # everything defined w.r.t. degrees
    "radians": 180.0 / np.pi, # everything normalized to degrees
}

distance_scale_factors = {
    "meters": 1.0, # everything defined w.r.t. meters,
    "kilometers": 1e3,
    "milimeters": 1e-3, # accounting for soundata typo
    "millimeters": 1e-3,
    "centimeters": 1e-2,
}


def fdiv(x, y) -> float:
    return float(Decimal(str(x)) / Decimal(str(y)))


def fmod(x, y) -> float:
    return float(Decimal(str(x)) % Decimal(str(y)))


def convert_units(val, input_units, target_units, scale_factor_dict) -> float:
    inp_factor = scale_factor_dict[target_units]
    out_factor = scale_factor_dict[input_units]
    factor =  fdiv(inp_factor, out_factor)
    return val * factor


def norm_time(time, input_units, target_units=TIME_UNITS) -> float:
    return convert_units(
        val=time,
        input_units=input_units,
        target_units=target_units,
        scale_factor_dict=time_scale_factors)


def norm_dist(dist, input_units, target_units=DISTANCE_UNITS) -> float:
    return convert_units(
        val=dist,
        input_units=input_units,
        target_units=target_units,
        scale_factor_dict=distance_scale_factors)


def norm_angle(angle, input_units, target_units=ANGLE_UNITS) -> float:
    angle = convert_units(
        val=angle,
        input_units=input_units,
        target_units=target_units,
        scale_factor_dict=angle_scale_factors)

    low = 0.0
    high = fdiv(360.0, angle_scale_factors[target_units])
    if not (low <= angle < high):
        # ensure in [0, 360.0) or [0, 2pi)
        angle = fmod(angle, high)

    return angle