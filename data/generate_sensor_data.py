"""
Generates synthetic manufacturing sensor/machine log data.

Design goal: this is NOT random noise. Each machine gets a simulated life history
where a subset of machines drift toward failure (rising vibration, rising
temperature, more frequent short stops) over the ~30 days before the failure
event, while healthy machines fluctuate around a stable baseline. This gives a
downstream classifier real signal to learn, and gives us something concrete to
talk about in the README ("what does an engineered failure pattern look like").

Output: data/raw/sensor_logs.csv
Columns:
    machine_id       - e.g. M001
    machine_type     - e.g. CNC_MILL, CONVEYOR, PRESS, ROBOT_ARM, PUMP
    timestamp        - daily reading
    temperature_c    - operating temperature in Celsius
    vibration_mm_s   - vibration velocity in mm/s (ISO 10816-ish units)
    runtime_hours     - cumulative runtime hours as of this reading
    days_since_maintenance - days since last maintenance event
    failed_within_7d - label: 1 if the machine fails within the next 7 days, else 0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)

MACHINE_TYPES = ["CNC_MILL", "CONVEYOR", "PRESS", "ROBOT_ARM", "PUMP"]

# Baseline "healthy" operating ranges per machine type (mean, std)
BASELINES = {
    "CNC_MILL":  {"temp": (55, 3), "vib": (2.0, 0.3)},
    "CONVEYOR":  {"temp": (40, 2), "vib": (1.2, 0.2)},
    "PRESS":     {"temp": (65, 4), "vib": (3.0, 0.4)},
    "ROBOT_ARM": {"temp": (48, 3), "vib": (1.5, 0.25)},
    "PUMP":      {"temp": (60, 3), "vib": (2.5, 0.35)},
}

N_MACHINES = 60
DAYS_OF_HISTORY = 180
FAILURE_WINDOW_DAYS = 7      # label horizon: "will it fail in the next 7 days"
DEGRADE_DAYS = 30            # how long the pre-failure drift lasts
PCT_MACHINES_THAT_FAIL = 0.35  # ~35% of machines experience at least one failure event


def simulate_machine(machine_id: str, machine_type: str, start_date: datetime):
    temp_mean, temp_std = BASELINES[machine_type]["temp"]
    vib_mean, vib_std = BASELINES[machine_type]["vib"]

    will_fail = np.random.rand() < PCT_MACHINES_THAT_FAIL
    failure_day = None
    if will_fail:
        # Failure happens sometime after the degrade window, leaving room for a
        # "before degradation" period as well so the model sees contrast.
        failure_day = np.random.randint(DEGRADE_DAYS + 10, DAYS_OF_HISTORY - 5)

    records = []
    runtime_hours = np.random.uniform(500, 5000)  # machines start with varied prior usage
    days_since_maint = np.random.randint(0, 20)

    for day in range(DAYS_OF_HISTORY):
        ts = start_date + timedelta(days=day)

        # Determine drift factor: 0 = healthy baseline, 1 = at failure
        drift = 0.0
        if will_fail and failure_day - DEGRADE_DAYS <= day < failure_day:
            drift = (day - (failure_day - DEGRADE_DAYS)) / DEGRADE_DAYS  # 0 -> 1 ramp

        # Degrading machines trend up in temp/vibration; healthy machines stay flat
        temp = np.random.normal(temp_mean + drift * temp_mean * 0.35, temp_std * (1 + drift))
        vib = np.random.normal(vib_mean + drift * vib_mean * 0.9, vib_std * (1 + drift))

        runtime_hours += np.random.uniform(6, 20)  # ~6-20 operating hrs/day
        days_since_maint += 1

        # Maintenance event resets the clock periodically (independent of failure)
        if days_since_maint > np.random.randint(25, 45):
            days_since_maint = 0

        # Label: will this machine fail within the next FAILURE_WINDOW_DAYS?
        label = 0
        if will_fail and failure_day - FAILURE_WINDOW_DAYS <= day < failure_day:
            label = 1

        records.append({
            "machine_id": machine_id,
            "machine_type": machine_type,
            "timestamp": ts.strftime("%Y-%m-%d"),
            "temperature_c": round(max(temp, 0), 2),
            "vibration_mm_s": round(max(vib, 0), 3),
            "runtime_hours": round(runtime_hours, 1),
            "days_since_maintenance": days_since_maint,
            "failed_within_7d": label,
        })

        # Machine is taken offline the day it "fails"; stop generating after that
        if will_fail and day == failure_day:
            break

    return records


def main():
    start_date = datetime(2025, 1, 1)
    all_records = []
    for i in range(1, N_MACHINES + 1):
        machine_id = f"M{i:03d}"
        machine_type = MACHINE_TYPES[(i - 1) % len(MACHINE_TYPES)]
        all_records.extend(simulate_machine(machine_id, machine_type, start_date))

    df = pd.DataFrame(all_records)
    df.to_csv("data/raw/sensor_logs.csv", index=False)

    print(f"Generated {len(df)} rows across {N_MACHINES} machines.")
    print(f"Positive label rate (failed_within_7d=1): {df['failed_within_7d'].mean():.3%}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
