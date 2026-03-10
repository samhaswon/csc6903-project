"""
Performs a rough grid simulation.
"""
from dataclasses import dataclass
import sys
import time

# External packages to install. Commands (Windows):
# python3 -m venv venv
# venv/scripts/activate
# pip install matplotlib numba numpy
import matplotlib.pyplot as plt
from numba import njit, jit
import numpy as np

# -----------------------------
# Global simulation parameters
# -----------------------------
TICK_TARGET_SEC = 0.0         # 0.0 => run as fast as possible
TICKS_PER_DAY = 2400          # day length in ticks
TICK_TARGET_TOTAL = 2400 * 7  # 0 for unbounded simulation
SIM_DT_SEC = 0.25  # simulated seconds per tick when running "as fast as possible"
PRINT_N_TICKS = 100  # Print only after a certain number of ticks. 1 is every tick
RNG_SEED = 1337

# Blackout thresholds (tune)
IMBALANCE_ABS_MW = 15.0
IMBALANCE_FRAC = 0.12

# Branch constraints
BRANCH_OVER_CAPACITY_BLACKOUT = True

# Generation behavior (ramp-aware dispatch below)
LOSS_BIAS = 1.01
DEADBAND_MW = 0.25

# -----------------------------
# Global state
# -----------------------------
RNG = np.random.default_rng(RNG_SEED)

# Dispatchable generators
# Gen 0 basically has to kick-start the simulation, so keep it high
GEN_CAPACITY_MW = np.array([5.0, 6.0, 4.0], dtype=np.float64)
GEN_OUTPUT_MW = np.array([5.0, 3.0, 4.0], dtype=np.float64)
GEN_MAX_RAMP_MW_PER_SEC = np.array([65.0, 6.0, 0.5], dtype=np.float64)
GEN_COST_PER_MWH = np.array([71.0, 44.0, 32.0], dtype=np.float64)

# Solar
SOLAR_CAPACITY_MW = 4.0
SOLAR_DAY_FACTOR = 1.0
SOLAR_DAY_PEAK_SHIFT = 0.0
SOLAR_DAY_WIDTH = 0.16
SOLAR_CLOUD_LPF = 1.0
SOLAR_EVENT_ACTIVE = False
SOLAR_EVENT_TICKS_LEFT = 0
SOLAR_EVENT_FACTOR = 0.85  # multiplies solar during an event

# Branch caps (MW), pick values that make overload possible but not constant
BRANCH_CAP_MW = np.array(
    [
        38.0, 38.2, 39.8, 38.4, 39.0, 38.1, 38.2, 38.3,
    ],
    dtype=np.float64)

# Loads: 1000 customers arranged as 2D (branches x customers)
N_BRANCHES = int(BRANCH_CAP_MW.shape[0])
N_CUSTOMERS_TOTAL = 24000
N_CUSTOMERS_PER_BRANCH = N_CUSTOMERS_TOTAL // N_BRANCHES

if N_CUSTOMERS_PER_BRANCH * N_BRANCHES != N_CUSTOMERS_TOTAL:
    raise ValueError("N_CUSTOMERS_TOTAL must be divisible by N_BRANCHES")

# Per-customer base load in kW (continuous)
BASE_KW = np.clip(
    RNG.normal(loc=0.9, scale=0.35, size=(N_BRANCHES, N_CUSTOMERS_PER_BRANCH)),
    0.1,
    2.5,
).astype(np.float64)

# HVAC and water heater states (0/1)
HVAC_ON = np.zeros((N_BRANCHES, N_CUSTOMERS_PER_BRANCH), dtype=np.int8)
WH_ON = np.zeros((N_BRANCHES, N_CUSTOMERS_PER_BRANCH), dtype=np.int8)

# Small per-customer preference factors for variability
HVAC_SENS = np.clip(RNG.normal(1.0, 0.12, size=(N_BRANCHES, N_CUSTOMERS_PER_BRANCH)), 0.7, 1.1)
WH_USAGE = np.clip(RNG.normal(1.0, 0.18, size=(N_BRANCHES, N_CUSTOMERS_PER_BRANCH)), 0.6, 1.3)

HVAC_MIN_ON_TICKS = 1    # e.g. 12 ticks * 0.25s = 3s simulated
HVAC_MIN_OFF_TICKS = 10
HVAC_TIMER = np.zeros((N_BRANCHES, N_CUSTOMERS_PER_BRANCH), dtype=np.int16)


@dataclass(frozen=True)
class Totals:
    total_load_mw: float
    total_gen_mw: float
    branch_loads_mw: np.ndarray
    solar_mw: float


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def daily_phase_from_tick(tick_idx: int) -> float:
    return (tick_idx % TICKS_PER_DAY) / float(TICKS_PER_DAY)


@njit(cache=True)
def outdoor_temp_base_c(day_phase: float) -> float:
    """
    Simple correlated outdoor temperature profile.
    - Low near early morning, high mid/late afternoon.
    """
    # Rough: 12C overnight low, 31C afternoon high, plus a mild evening shoulder
    temp = 21.5 + 9.5 * np.sin(2.0 * np.pi * (day_phase - 0.30))
    temp += 2.5 * np.exp(-((day_phase - 0.75) / 0.12) ** 2)
    return float(temp)


BRANCH_TEMP_OFFSETS_C = RNG.uniform(-0.3, 0.3, N_BRANCHES).astype(np.float64)

def outdoor_temps_by_branch_c(day_phase: float) -> np.ndarray:
    """
    Branch temps are similar, with small fixed offsets and small shared noise.
    """
    base = outdoor_temp_base_c(day_phase)
    if SOLAR_EVENT_ACTIVE:
        base -= 2.0
    shared_noise = float(RNG.normal(0.0, 0.25))
    per_branch_noise = RNG.normal(0.0, 0.15, size=N_BRANCHES)
    return (base + BRANCH_TEMP_OFFSETS_C + shared_noise + per_branch_noise).astype(np.float64)


def sample_daily_solar_state() -> None:
    global SOLAR_DAY_FACTOR, SOLAR_DAY_PEAK_SHIFT, SOLAR_DAY_WIDTH
    global SOLAR_CLOUD_LPF, SOLAR_EVENT_ACTIVE, SOLAR_EVENT_TICKS_LEFT, SOLAR_EVENT_FACTOR

    # Dramatic: lots of cloudy days
    cloudy = RNG.random() < 0.35
    if cloudy:
        SOLAR_DAY_FACTOR = float(RNG.uniform(0.10, 0.45))  # dramatic attenuation
    else:
        SOLAR_DAY_FACTOR = float(RNG.uniform(0.80, 1.10))

    SOLAR_DAY_PEAK_SHIFT = float(np.clip(RNG.normal(0.0, 0.025), -0.06, 0.06))
    SOLAR_DAY_WIDTH = float(np.clip(RNG.normal(0.16, 0.03), 0.10, 0.24))

    SOLAR_CLOUD_LPF = SOLAR_DAY_FACTOR

    SOLAR_EVENT_ACTIVE = False
    SOLAR_EVENT_TICKS_LEFT = 0
    SOLAR_EVENT_FACTOR = 0.85


def update_intraday_cloudiness(tick_idx_in_day: int) -> None:
    """
    Dramatic cloudiness:
    - slow drift around daily factor
    - occasional cloud events (big temporary drop) that persist many ticks
    """
    global SOLAR_CLOUD_LPF, SOLAR_EVENT_ACTIVE, SOLAR_EVENT_TICKS_LEFT, SOLAR_EVENT_FACTOR

    # Base slow drift (keeps it from being perfectly flat)
    drift_noise = float(RNG.normal(0.0, 0.02))
    SOLAR_CLOUD_LPF = 0.990 * SOLAR_CLOUD_LPF + 0.010 * SOLAR_DAY_FACTOR + drift_noise
    SOLAR_CLOUD_LPF = float(np.clip(SOLAR_CLOUD_LPF, 0.05, 1.15))

    # Only do events during daylight-ish hours
    day_phase = tick_idx_in_day / float(TICKS_PER_DAY)
    daylight = 0.18 < day_phase < 0.82

    if daylight and not SOLAR_EVENT_ACTIVE:
        # Start events sometimes, more on cloudy days
        p_start = 0.002 + 0.010 * (1.0 - min(SOLAR_DAY_FACTOR, 1.0))
        if RNG.random() < p_start:
            SOLAR_EVENT_ACTIVE = True
            # Duration: 2%..10% of the day
            SOLAR_EVENT_TICKS_LEFT = int(RNG.integers(int(0.02 * TICKS_PER_DAY), int(0.10 * TICKS_PER_DAY)))
            # Event factor: deep dip
            SOLAR_EVENT_FACTOR = float(RNG.uniform(0.40, 0.60))

    if SOLAR_EVENT_ACTIVE:
        SOLAR_EVENT_TICKS_LEFT -= 1
        if SOLAR_EVENT_TICKS_LEFT <= 0:
            SOLAR_EVENT_ACTIVE = False
            SOLAR_EVENT_FACTOR = 0.85


@jit(forceobj=True, looplift=True, cache=True)
def solar_output_mw(day_phase: float) -> float:
    # Peak centered at noon with daily shift
    center = 0.5 + SOLAR_DAY_PEAK_SHIFT
    width = SOLAR_DAY_WIDTH

    shape = np.exp(-((day_phase - center) / width) ** 2)
    shape = float(shape if shape > 0.02 else 0.0)

    tick_var = float(np.clip(RNG.normal(1.0, 0.01), 0.95, 1.05))

    event = SOLAR_EVENT_FACTOR if SOLAR_EVENT_ACTIVE else 1.0
    mw = SOLAR_CAPACITY_MW * shape * SOLAR_CLOUD_LPF * event * tick_var
    return float(np.clip(mw, 0.0, SOLAR_CAPACITY_MW))


def update_customer_base_load_kw(day_phase: float) -> None:
    """
    Update continuous base loads (kW) with a daily trend and some noise.
    Keeps values in [0.1, ~2.5] kW per customer.
    """
    # Daily trend factor for base usage
    base_factor = 0.45 + 0.25 * np.sin(2.0 * np.pi * (day_phase - 0.25))
    base_factor += 0.38 * np.exp(-((day_phase - 0.80) / 0.14) ** 2)  # evening activity
    base_factor = clamp(base_factor, 0.65, 1.25)

    noise = RNG.normal(0.0, 0.03, size=BASE_KW.shape)
    BASE_KW[:] = np.clip(BASE_KW * base_factor + noise, 0.1, 2.0)


@jit(forceobj=True, looplift=True, cache=True)
def hvac_transition_probs(temp_c: float) -> tuple[float, float]:
    """
    Return (p_on_if_off, p_off_if_on) for HVAC.
    Higher temp => HVAC more likely to turn on and less likely to turn off.
    """
    on_threshold = 25.0
    off_threshold = 20.5

    heat = clamp((temp_c - on_threshold) / 300.0, 0.0, 0.5)
    cool = clamp((off_threshold - temp_c) / 100.0, 0.0, 0.5)

    p_on = 0.005 + 0.12 * heat
    p_off = 0.2 + 0.18 * cool
    return p_on, p_off


@jit(forceobj=True, looplift=True, cache=True)
def water_heater_transition_probs(day_phase: float) -> tuple[float, float]:
    """
    Return (p_on_if_off, p_off_if_on) for water heater.
    Mostly intermittent, with end-of-day bump.
    """
    morning_bump = np.exp(-((day_phase - 0.25) / 0.08) ** 2)
    evening_bump = np.exp(-((day_phase - 0.70) / 0.10) ** 2)

    p_on = 0.004 + 0.010 * float(morning_bump) + 0.045 * float(evening_bump)
    p_off = 0.78
    return clamp(p_on, 0.0, 0.20), clamp(p_off, 0.05, 1.0)


def update_hvac_and_wh_states(day_phase: float, temps_c: np.ndarray) -> None:
    """
    Update HVAC (+7kW) and water heater (+5kW) discrete states.
    """
    # HVAC per-branch probabilities based on correlated temperature
    for b in range(N_BRANCHES):
        p_on, p_off = hvac_transition_probs(float(temps_c[b]))

        # Individualize a bit
        p_on_mat = np.clip(p_on * HVAC_SENS[b], 0.0, 0.15)
        p_off_mat = np.clip(p_off / HVAC_SENS[b], 0.01, 0.80)

        rand = RNG.random(size=HVAC_ON.shape[1])
        turning_on = (HVAC_ON[b] == 0) & (rand < p_on_mat)
        rand2 = RNG.random(size=HVAC_ON.shape[1])
        turning_off = (HVAC_ON[b] == 1) & (rand2 < p_off_mat)

        HVAC_ON[b, turning_on] = 1
        HVAC_ON[b, turning_off] = 0

    # Water heater globally influenced by day phase
    p_on, p_off = water_heater_transition_probs(day_phase)
    p_on_mat = np.clip(p_on * WH_USAGE, 0.0, 0.25)
    p_off_mat = np.clip(p_off / WH_USAGE, 0.05, 0.70)

    r_on = RNG.random(size=WH_ON.shape)
    r_off = RNG.random(size=WH_ON.shape)

    turning_on = (WH_ON == 0) & (r_on < p_on_mat)
    turning_off = (WH_ON == 1) & (r_off < p_off_mat)

    WH_ON[turning_on] = 1
    WH_ON[turning_off] = 0


@jit(forceobj=True, looplift=True, cache=True)
def compute_branch_loads_mw(day_phase: float) -> np.ndarray:
    """
    Compute branch loads in MW from base + discrete appliances + noise.
    Enforces per-customer cap of 14 kW.
    """
    update_customer_base_load_kw(day_phase)
    temps_c = outdoor_temps_by_branch_c(day_phase)
    update_hvac_and_wh_states(day_phase, temps_c)

    # kW: base + 7kW HVAC + 2kW water heater
    kw = BASE_KW + 7.0 * HVAC_ON + 2.0 * WH_ON

    # Small measurement / behavior noise, but keep discrete contributions intact-ish
    kw += RNG.normal(0.0, 0.05, size=kw.shape)

    # Clamp each customer to [0.1, 14] kW
    kw = np.clip(kw, 0.1, 14.0)

    # Sum per branch and convert to MW
    branch_mw = kw.sum(axis=1) / 1000.0
    return branch_mw.astype(np.float64)


@njit(cache=True)
def ramp_feasible_bounds(gen_output_mw: np.ndarray, dt_sec: float) -> tuple[np.ndarray, np.ndarray]:
    max_delta = GEN_MAX_RAMP_MW_PER_SEC * dt_sec
    min_out = np.maximum(0.0, gen_output_mw - max_delta)
    max_out = np.minimum(GEN_CAPACITY_MW, gen_output_mw + max_delta)
    return min_out, max_out


def ramp_constrained_economic_dispatch(
    gen_output_mw: np.ndarray,
    demand_mw: float,
    dt_sec: float,
) -> np.ndarray:
    demand_mw = float(max(0.0, demand_mw))

    min_out, max_out = ramp_feasible_bounds(gen_output_mw, dt_sec)

    fleet_min = float(min_out.sum())
    fleet_max = float(max_out.sum())

    if demand_mw <= fleet_min:
        return min_out.copy()

    if demand_mw >= fleet_max:
        return max_out.copy()

    out = min_out.copy()
    remaining = float(demand_mw - out.sum())

    order = np.argsort(GEN_COST_PER_MWH)  # cheapest first
    for i in order:
        headroom = float(max_out[i] - out[i])
        take = min(headroom, remaining)
        out[i] += take
        remaining -= take
        if remaining <= 1e-9:
            break

    return out


def update_generation(gen_output_mw: np.ndarray, load_mw: float, solar_mw: float, dt_sec: float) -> np.ndarray:
    net_load = max(0.0, float(load_mw - solar_mw))
    desired_dispatchable = net_load * LOSS_BIAS

    current_dispatchable = float(gen_output_mw.sum())
    if abs(desired_dispatchable - current_dispatchable) < DEADBAND_MW:
        desired_dispatchable = current_dispatchable

    return ramp_constrained_economic_dispatch(gen_output_mw, desired_dispatchable, dt_sec)


def totals(branch_loads_mw: np.ndarray, gen_output_mw: np.ndarray, solar_mw: float) -> Totals:
    total_load = float(branch_loads_mw.sum())
    total_gen = float(gen_output_mw.sum() + solar_mw)
    return Totals(
        total_load_mw=total_load,
        total_gen_mw=total_gen,
        branch_loads_mw=branch_loads_mw.copy(),
        solar_mw=float(solar_mw),
    )


def check_blackout(t: Totals) -> None:
    eps = 1e-9
    imbalance = t.total_gen_mw - t.total_load_mw
    abs_imbalance = abs(imbalance)
    frac_limit = IMBALANCE_FRAC * max(t.total_load_mw, eps)

    if abs_imbalance > IMBALANCE_ABS_MW or abs_imbalance > frac_limit:
        direction = "over-generation" if imbalance > 0 else "under-generation"
        print(
            f"BLACKOUT: imbalance too large ({direction}). "
            f"gen={t.total_gen_mw:.2f} MW load={t.total_load_mw:.2f} MW diff={imbalance:+.2f} MW"
        )
        sys.exit(2)

    over_caps = t.branch_loads_mw - BRANCH_CAP_MW
    if BRANCH_OVER_CAPACITY_BLACKOUT and np.any(over_caps > 0.0):
        idx = int(np.argmax(over_caps))
        print(
            "BLACKOUT: branch overload. "
            f"branch={idx} load={t.branch_loads_mw[idx]:.2f} MW cap={BRANCH_CAP_MW[idx]:.2f} MW"
        )
        sys.exit(3)


def format_status(tick_idx: int, elapsed_s: float, t: Totals, gen_output_mw: np.ndarray) -> str:
    branch_str = " ".join(
        f"B{i}:{t.branch_loads_mw[i]:4.2f}/{BRANCH_CAP_MW[i]:.1f}"
        for i in range(len(t.branch_loads_mw))
    )
    gens_str = " ".join(f"G{i}:{gen_output_mw[i]:5.2f}" for i in range(gen_output_mw.size))
    return (
        f"tick={tick_idx:6d} t={elapsed_s:7.2f}s  "
        f"load={t.total_load_mw:6.2f} MW  gen={t.total_gen_mw:6.2f} MW  "
        f"diff={(t.total_gen_mw - t.total_load_mw):+6.2f} MW  "
        f"solar={t.solar_mw:5.2f}  {gens_str}  {branch_str}"
    )


def plot_results(history: dict) -> None:
    if not history["t"]:
        print("No history collected, skipping plot.")
        return

    t = np.array(history["t"], dtype=np.float64)[4:]
    load = np.array(history["load"], dtype=np.float64)[4:]
    total = np.array(history["total_gen"], dtype=np.float64)[4:]
    solar = np.array(history["solar"], dtype=np.float64)[4:]
    gens = np.array(history["gens"], dtype=np.float64)[4:]

    plt.figure(figsize=(10, 6))
    plt.plot(t, load, label="Load (total)")
    plt.plot(t, total, label="Total generation")

    for i in range(gens.shape[1]):
        plt.plot(t, gens[:, i], label=f"Gen {i}")

    plt.plot(t, solar, label="Solar")

    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.title("Grid simulation outputs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    global GEN_OUTPUT_MW

    history = {
        "t": [],
        "load": [],
        "total_gen": [],
        "solar": [],
        "gens": [],
    }

    print("Starting grid simulation. Ctrl+C to stop.")
    start = time.monotonic()
    last_tick_time = start

    exit_code = 0
    stop_reason = "stopped"

    tick_idx = 0

    try:
        while TICK_TARGET_TOTAL == 0 or tick_idx < TICK_TARGET_TOTAL:
            # Rate-control via sleep to target seconds per tick
            now = time.monotonic()
            dt_target = float(TICK_TARGET_SEC)

            if dt_target > 0.0:
                sleep_for = dt_target - (now - last_tick_time)
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                dt_sec = dt_target  # physics uses the target
            else:
                dt_sec = SIM_DT_SEC  # physics uses fixed simulated time
                # no sleeping, run as fast as possible

            last_tick_time = now
            elapsed_s = now - start  # real wall clock for logging

            if tick_idx % TICKS_PER_DAY == 0:
                sample_daily_solar_state()

            day_phase = daily_phase_from_tick(tick_idx)

            tick_in_day = tick_idx % TICKS_PER_DAY
            if tick_in_day == 0:
                sample_daily_solar_state()

            update_intraday_cloudiness(tick_in_day)
            solar_mw = solar_output_mw(day_phase)

            branch_loads_mw = compute_branch_loads_mw(day_phase)
            solar_mw = solar_output_mw(day_phase)

            t_before = totals(branch_loads_mw, GEN_OUTPUT_MW, solar_mw)
            GEN_OUTPUT_MW = update_generation(GEN_OUTPUT_MW, t_before.total_load_mw, solar_mw, dt_sec)
            t_after = totals(branch_loads_mw, GEN_OUTPUT_MW, solar_mw)

            if tick_idx % PRINT_N_TICKS == 0:
                print(format_status(tick_idx, elapsed_s, t_after, GEN_OUTPUT_MW))

            history["t"].append(elapsed_s)
            history["load"].append(t_after.total_load_mw)
            history["total_gen"].append(t_after.total_gen_mw)
            history["solar"].append(t_after.solar_mw)
            history["gens"].append(GEN_OUTPUT_MW.copy())

            check_blackout(t_after)

            tick_idx += 1

    except KeyboardInterrupt:
        stop_reason = "keyboard interrupt"
        exit_code = 0
    except SystemExit as exc:
        stop_reason = "system exit"
        exit_code = int(exc.code) if exc.code is not None else 1
    finally:
        print(f"Simulation {stop_reason}. Plotting results...")
        try:
            plot_results(history)
        finally:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
