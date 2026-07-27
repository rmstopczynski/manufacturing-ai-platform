"""
Generates the synthetic unstructured knowledge base: equipment manuals, SOPs, and
maintenance log entries for the 5 machine types used in the structured dataset
(CNC_MILL, CONVEYOR, PRESS, ROBOT_ARM, PUMP).

Design choice: these are hand-authored templates with per-type variable substitution,
not LLM-generated or scraped. That's deliberate — it keeps content grounded in the same
failure patterns (rising vibration/temperature) baked into the sensor data, so retrieval
+ the ML prediction actually reinforce each other in the RAG answers (e.g. a maintenance
log entry that describes "vibration climbed for two weeks before bearing failure" should
be retrievable when a machine's live sensor trend looks exactly like that).

Each document is written to rag/documents/<doc_id>.txt with a small metadata header
(parsed back out at ingestion time) so Chroma can filter/attribute retrieval results by
doc_type and machine_type.

Output: rag/documents/*.txt + rag/documents/manifest.json
"""

import json
import os

OUT_DIR = "rag/documents"
os.makedirs(OUT_DIR, exist_ok=True)

MACHINE_TYPES = ["CNC_MILL", "CONVEYOR", "PRESS", "ROBOT_ARM", "PUMP"]

DISPLAY_NAMES = {
    "CNC_MILL": "CNC Milling Machine",
    "CONVEYOR": "Conveyor System",
    "PRESS": "Hydraulic Press",
    "ROBOT_ARM": "Robotic Arm",
    "PUMP": "Industrial Pump",
}

NORMAL_RANGES = {
    "CNC_MILL": "operating temperature 50-60°C, vibration 1.5-2.5 mm/s",
    "CONVEYOR": "operating temperature 36-44°C, vibration 0.9-1.5 mm/s",
    "PRESS": "operating temperature 58-72°C, vibration 2.4-3.6 mm/s",
    "ROBOT_ARM": "operating temperature 42-54°C, vibration 1.1-1.9 mm/s",
    "PUMP": "operating temperature 54-66°C, vibration 2.0-3.0 mm/s",
}

FAILURE_MODES = {
    "CNC_MILL": "spindle bearing wear",
    "CONVEYOR": "drive belt misalignment",
    "PRESS": "hydraulic seal degradation",
    "ROBOT_ARM": "joint actuator wear",
    "PUMP": "impeller cavitation and bearing wear",
}

documents = []
doc_counter = 1


def add_doc(doc_type, machine_type, title, body):
    global doc_counter
    doc_id = f"doc_{doc_counter:03d}"
    doc_counter += 1
    documents.append({
        "doc_id": doc_id,
        "doc_type": doc_type,      # manual | sop | maintenance_log
        "machine_type": machine_type,  # one of MACHINE_TYPES, or "GENERAL"
        "title": title,
        "body": body.strip(),
    })


# ---------------------------------------------------------------------------
# Equipment manuals — one per machine type
# ---------------------------------------------------------------------------
for mt in MACHINE_TYPES:
    name = DISPLAY_NAMES[mt]
    rng = NORMAL_RANGES[mt]
    failure = FAILURE_MODES[mt]
    add_doc("manual", mt, f"{name} — Equipment Manual (Section 4: Operating Parameters)", f"""
{name} Equipment Manual
Section 4: Operating Parameters and Warning Signs

Normal operating range: {rng}. Readings outside this range for more than a few
consecutive readings should be logged and reviewed at the next maintenance check.

The most common failure mode for this equipment class is {failure}. Early indicators
include a gradual upward trend in vibration readings over 2-4 weeks, increasing
day-to-day variability in temperature (not just a higher average, but less
consistency reading to reading), and audible changes in operating noise.

If vibration readings show a sustained upward trend combined with rising temperature
volatility, schedule an inspection within 7 days even if individual readings remain
within nominal range — the trend itself is the warning sign, not any single reading.

Do not wait for a hard threshold breach before inspecting. Machines in this class have
historically failed while individual sensor readings were still inside the "normal"
range, because the drift toward failure happens gradually enough that no single
reading looks alarming in isolation.
""")

# ---------------------------------------------------------------------------
# SOPs — general procedures, not machine-type specific (except one troubleshooting SOP per type)
# ---------------------------------------------------------------------------
add_doc("sop", "GENERAL", "SOP-001: Lockout/Tagout Procedure Before Maintenance", """
SOP-001: Lockout/Tagout (LOTO) Procedure

Before performing any maintenance on plant equipment:
1. Notify all affected personnel that maintenance is beginning.
2. Identify all energy sources (electrical, hydraulic, pneumatic) for the equipment.
3. Shut down the equipment using the normal stop procedure.
4. Isolate each energy source and apply a lock and tag at each isolation point.
5. Verify zero energy state by attempting a normal start (with lock in place) and
   confirming the equipment does not activate.
6. Perform the required maintenance.
7. After maintenance, remove tools, reinstall guards, and confirm the area is clear
   before removing locks and tags in reverse order.

No maintenance task, however brief, is exempt from this procedure. Skipping LOTO for
"quick" fixes is the single most common root cause of maintenance-related injuries
plant-wide.
""")

add_doc("sop", "GENERAL", "SOP-002: Routine Maintenance Scheduling", """
SOP-002: Routine Maintenance Scheduling

Standard maintenance intervals are set at 30-45 days for most rotating and hydraulic
equipment, tracked via days-since-last-maintenance. Equipment approaching or exceeding
40 days since last maintenance should be prioritized for the next scheduled window,
even if sensor readings currently look nominal — maintenance intervals are a
preventive measure, not a reactive one.

When a machine is flagged by the predictive maintenance system as elevated-risk,
its maintenance should be moved up regardless of where it falls in the normal
rotation, and the technician should be given the specific sensor trend that
triggered the flag (not just "elevated risk") so they know what to inspect first.
""")

for mt in MACHINE_TYPES:
    name = DISPLAY_NAMES[mt]
    failure = FAILURE_MODES[mt]
    add_doc("sop", mt, f"SOP-Troubleshooting-{mt}: Responding to Elevated Vibration or Temperature", f"""
SOP-Troubleshooting-{mt}: Responding to Elevated Vibration/Temperature Alerts on {name}s

1. Confirm the alert against the live sensor dashboard — check whether the trend is
   isolated to one reading (likely a sensor glitch) or sustained over multiple days
   (likely a real mechanical issue).
2. Cross-reference days-since-last-maintenance. A machine well past its maintenance
   window showing a sustained trend is higher priority than one recently serviced.
3. The dominant failure mode for {name}s is {failure}. Inspect the corresponding
   component first before broader diagnostics.
4. If the trend matches the pre-failure pattern described in the equipment manual
   (rising vibration, increasing temperature variability), schedule inspection within
   7 days per SOP-002, and follow SOP-001 (LOTO) before any hands-on inspection.
5. Log findings in the maintenance log regardless of outcome, including cases where
   inspection found no issue — this data improves future predictions.
""")

# ---------------------------------------------------------------------------
# Maintenance log entries — short, varied, some describe real failures, some describe
# false alarms / no-issue-found inspections (important for RAG realism)
# ---------------------------------------------------------------------------
log_entries = [
    ("CNC_MILL", "Spindle bearing replaced after 3-week vibration climb",
     "Machine flagged for rising vibration trend starting approx. 3 weeks prior. "
     "Inspection found spindle bearing wear consistent with the manual's described "
     "failure pattern. Bearing replaced, vibration returned to baseline (2.1 mm/s) "
     "within 24 hours of restart."),
    ("CNC_MILL", "False alarm — sensor recalibration, no mechanical issue found",
     "Vibration sensor showed an isolated spike on a single reading with no sustained "
     "trend before or after. Inspection found no mechanical issue. Sensor was "
     "recalibrated as a precaution. Logging as false alarm per SOP-002 guidance to "
     "record no-issue-found inspections."),
    ("CONVEYOR", "Drive belt misalignment corrected during scheduled maintenance",
     "Routine 40-day maintenance check found early-stage belt misalignment, not yet "
     "reflected in a strong sensor trend. Belt realigned and tensioned. Flagged as a "
     "good example of catching an issue before it produced a measurable sensor drift."),
    ("CONVEYOR", "Belt failure after delayed maintenance window",
     "Machine was 52 days since last maintenance (12 days past the standard window) "
     "when vibration trend crossed the elevated-risk threshold. Inspection found "
     "advanced belt wear; belt failed completely two days after the alert during "
     "attempted continued operation. Replaced belt and drive pulley. Recommend "
     "stricter enforcement of the 40-day maintenance window going forward."),
    ("PRESS", "Hydraulic seal replaced following gradual temperature rise",
     "Temperature trend showed the characteristic rolling-average rise over "
     "approximately 3 weeks with increasing day-to-day variability. Hydraulic seal "
     "inspection confirmed degradation. Seal replaced, temperature volatility "
     "dropped back to typical range within one day."),
    ("PRESS", "Unexplained vibration spike, root cause not confirmed",
     "Single-day vibration spike with no clear trend before or after. Inspection did "
     "not conclusively identify a cause; loose mounting bolt was tightened as a "
     "precaution. Recommend monitoring for recurrence before ruling out a developing "
     "seal issue."),
    ("ROBOT_ARM", "Joint actuator wear identified and replaced",
     "Sustained vibration increase over 2.5 weeks matched the documented joint "
     "actuator wear pattern for this machine class. Actuator on joint 3 replaced. "
     "Post-replacement vibration returned to baseline."),
    ("ROBOT_ARM", "No issue found on flagged machine, continued monitoring recommended",
     "Machine flagged for a mild vibration trend that had not yet crossed the "
     "sustained-drift threshold described in the manual. Inspection found components "
     "within tolerance. No action taken; flagged for closer monitoring on next cycle "
     "rather than immediate part replacement."),
    ("PUMP", "Impeller cavitation causing early bearing wear",
     "Rising vibration combined with elevated temperature variability over 3 weeks. "
     "Inspection found impeller cavitation damage, which had begun accelerating "
     "bearing wear. Impeller and bearing both replaced in the same maintenance window "
     "to avoid a second shutdown."),
    ("PUMP", "Routine maintenance, no anomalies",
     "Standard 35-day maintenance check. Sensor trends nominal throughout the prior "
     "interval. No component wear beyond expected levels. No action needed beyond "
     "standard lubrication per manual."),
    ("CNC_MILL", "Delayed maintenance nearly caused unplanned downtime",
     "Machine had gone 48 days since last maintenance (past the 45-day upper bound in "
     "SOP-002) when a strong vibration trend was flagged. Inspection found advanced "
     "spindle bearing wear, close to the failure threshold described in the manual. "
     "Bearing replaced just ahead of what likely would have been an unplanned failure."),
    ("CONVEYOR", "Sensor malfunction produced false rising-temperature trend",
     "Apparent temperature drift over several days was traced to a faulty temperature "
     "sensor reading progressively higher due to a loose connection, not an actual "
     "mechanical issue. Connection repaired, sensor recalibrated. Important reminder "
     "that not every sensor trend reflects a real mechanical trend — cross-check "
     "vibration alongside temperature before concluding a real issue is developing."),
]

for mt, title, body in log_entries:
    add_doc("maintenance_log", mt, title, body)


with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(documents, f, indent=2)

for doc in documents:
    with open(os.path.join(OUT_DIR, f"{doc['doc_id']}.txt"), "w") as f:
        f.write(f"[doc_type: {doc['doc_type']}] [machine_type: {doc['machine_type']}]\n")
        f.write(f"Title: {doc['title']}\n\n")
        f.write(doc["body"])

print(f"Generated {len(documents)} documents in {OUT_DIR}/")
by_type = {}
for d in documents:
    by_type[d["doc_type"]] = by_type.get(d["doc_type"], 0) + 1
print(by_type)
