"""Script to create sample data Excel files for testing."""

import pandas as pd
from datetime import datetime, timedelta
import random

# Sample lessons learned data
SAMPLE_LESSONS = [
    {
        "lesson_id": "LL-001",
        "title": "Pump Seal Failure Due to Improper Installation",
        "description": "Centrifugal pump P-101 experienced seal failure within 24 hours of maintenance. Investigation revealed the mechanical seal was installed without proper lubrication of the O-rings, causing immediate wear and leakage.",
        "root_cause": "Mechanical seal O-rings were installed dry without proper lubricant. Maintenance procedure did not clearly specify lubrication requirements.",
        "corrective_action": "Updated maintenance procedure to include mandatory seal lubrication step. Added checklist item for seal installation verification. All pump seal replacements now require two-person verification.",
        "category": "mechanical",
        "equipment_tag": "P-101",
        "severity": "high",
        "date": "2024-03-15",
    },
    {
        "lesson_id": "LL-002",
        "title": "Heat Exchanger Tube Bundle Damage During Extraction",
        "description": "HX-205 tube bundle was damaged during turnaround extraction due to inadequate rigging assessment. The bundle weight exceeded crane capacity at the extraction angle, causing the bundle to strike the shell.",
        "root_cause": "Lift plan did not account for the additional weight of deposits inside tubes. No pre-extraction weight verification was performed.",
        "corrective_action": "Mandatory pre-extraction weight estimate for all tube bundles based on service history. Lift plans must include 20% safety margin. Required crane operator and rigger joint sign-off.",
        "category": "mechanical",
        "equipment_tag": "HX-205",
        "severity": "critical",
        "date": "2024-02-20",
    },
    {
        "lesson_id": "LL-003",
        "title": "Electrical Isolation Failure During Motor Maintenance",
        "description": "Maintenance technician received electrical shock while working on motor M-302 despite lockout/tagout being applied. Investigation found that the isolation was performed on the wrong breaker due to outdated electrical drawings.",
        "root_cause": "Electrical single-line diagrams had not been updated after a modification project. LOTO verification did not include voltage testing at the equipment.",
        "corrective_action": "Mandatory voltage testing at point of work before any electrical maintenance. All electrical drawings to be verified and updated during next turnaround. Added zero-energy verification step to LOTO procedure.",
        "category": "electrical",
        "equipment_tag": "M-302",
        "severity": "critical",
        "date": "2024-01-10",
    },
    {
        "lesson_id": "LL-004",
        "title": "Valve Packing Leak After Maintenance",
        "description": "Control valve V-450 developed packing leak immediately after maintenance. The packing was replaced but not properly compressed. Leak caused process upset and required emergency shutdown.",
        "root_cause": "Packing gland nuts were not torqued to specification. No torque values provided in the work pack.",
        "corrective_action": "Added specific torque values for all valve packing glands to maintenance procedures. Implemented torque verification checklist. Training provided on proper packing installation techniques.",
        "category": "mechanical",
        "equipment_tag": "V-450",
        "severity": "high",
        "date": "2024-04-05",
    },
    {
        "lesson_id": "LL-005",
        "title": "Compressor Vibration After Bearing Replacement",
        "description": "Compressor C-100 experienced high vibration after scheduled bearing replacement. Vibration levels exceeded alarm limits within first hour of operation. Unit had to be shut down for re-inspection.",
        "root_cause": "Bearing clearances were not verified after installation. Internal alignment check was skipped due to time pressure.",
        "corrective_action": "Mandatory post-installation vibration baseline before returning compressor to service. Added bearing clearance verification to procedure. No schedule pressure to skip critical alignment checks.",
        "category": "mechanical",
        "equipment_tag": "C-100",
        "severity": "high",
        "date": "2024-03-28",
    },
    {
        "lesson_id": "LL-006",
        "title": "Instrument Calibration Drift After Installation",
        "description": "Multiple pressure transmitters showed significant drift within weeks of installation. Investigation revealed transmitters were exposed to high ambient temperatures during storage before installation.",
        "root_cause": "Instruments were stored in non-climate-controlled warehouse. Temperature exposure caused calibration drift before installation.",
        "corrective_action": "Sensitive instruments must be stored in climate-controlled area. Pre-installation calibration verification required for all instruments stored more than 30 days. Added storage requirements to procurement specifications.",
        "category": "instrumentation",
        "equipment_tag": "PT-100",
        "severity": "medium",
        "date": "2024-02-15",
    },
    {
        "lesson_id": "LL-007",
        "title": "Tank Inspection Safety Incident - H2S Exposure",
        "description": "Worker entered tank T-500 for inspection and was exposed to residual H2S gas. Worker evacuated safely but required medical attention. Gas testing was performed but not at all required locations.",
        "root_cause": "Gas testing was only performed at tank manway, not at multiple levels inside tank. H2S had accumulated in low point of tank bottom.",
        "corrective_action": "Gas testing required at multiple elevations before and during confined space entry. Continuous gas monitoring mandatory for all confined space work. Enhanced confined space entry training for all personnel.",
        "category": "safety",
        "equipment_tag": "T-500",
        "severity": "critical",
        "date": "2024-01-25",
    },
    {
        "lesson_id": "LL-008",
        "title": "Flange Leak Due to Incorrect Gasket Material",
        "description": "High pressure flange on reactor R-200 developed leak during startup. Investigation found wrong gasket material was installed - standard spiral wound instead of high-temperature rated material.",
        "root_cause": "Gaskets were stored together without clear labeling. Work pack did not specify exact gasket part number. Visual identification was not possible between standard and high-temp gaskets.",
        "corrective_action": "All gaskets must be labeled with material specification and temperature rating. Work packs must include specific part numbers for all materials. Material verification sign-off added to flange assembly procedure.",
        "category": "mechanical",
        "equipment_tag": "R-200",
        "severity": "high",
        "date": "2024-04-15",
    },
    {
        "lesson_id": "LL-009",
        "title": "Coupling Failure Due to Misalignment",
        "description": "Pump P-205 coupling failed 48 hours after maintenance. Laser alignment was performed but not verified after coupling bolt torque. Bolt torquing caused alignment shift that exceeded coupling tolerance.",
        "root_cause": "Alignment verification was performed before final bolt torque. Torquing sequence created stress that shifted alignment beyond acceptable limits.",
        "corrective_action": "Final alignment verification must be performed after all bolt torquing is complete. Alignment tolerance reduced to 50% of coupling specification to allow for thermal growth. Added final alignment check to procedure.",
        "category": "mechanical",
        "equipment_tag": "P-205",
        "severity": "medium",
        "date": "2024-03-10",
    },
    {
        "lesson_id": "LL-010",
        "title": "Filter Element Installation Error",
        "description": "Lube oil filter F-150 was installed upside down, causing contamination of compressor lube oil system. Compressor bearings showed accelerated wear when inspected at next scheduled shutdown.",
        "root_cause": "Filter element design allowed installation in either direction. No visual indicator or mechanical interlock to prevent incorrect installation.",
        "corrective_action": "Replaced filter design with directional-keyed elements that cannot be installed incorrectly. Added installation orientation verification to filter replacement procedure. Training provided on critical lubrication system components.",
        "category": "mechanical",
        "equipment_tag": "F-150",
        "severity": "medium",
        "date": "2024-02-28",
    },
    {
        "lesson_id": "LL-011",
        "title": "Universal Lesson: Pre-Job Safety Brief Requirements",
        "description": "Multiple near-miss incidents occurred when workers began tasks without proper pre-job safety briefings. Analysis showed that rushed starts and assumed knowledge led to overlooked hazards.",
        "root_cause": "Pre-job safety briefs were being abbreviated or skipped during time-critical tasks. No formal verification that all workers understood task-specific hazards.",
        "corrective_action": "All work tasks require documented pre-job safety brief with sign-off from all participants. Brief must cover task-specific hazards, emergency procedures, and communication protocols. Applies to ALL maintenance work regardless of equipment type.",
        "category": "safety",
        "equipment_tag": None,
        "severity": "critical",
        "date": "2024-01-05",
    },
    {
        "lesson_id": "LL-012",
        "title": "Universal Lesson: Tool Control During Turnaround",
        "description": "Foreign object damage (FOD) incidents increased during turnaround due to lost tools. Multiple incidents of tools left inside equipment after maintenance completion.",
        "root_cause": "Tool control procedures were not consistently followed. Tool inventories not reconciled at job completion. Time pressure led to shortcuts in tool management.",
        "corrective_action": "Mandatory tool inventory at start and end of each shift. Tools must be tethered when working at height or over open equipment. Job cannot be signed off until tool count is verified. Applies universally to all maintenance activities.",
        "category": "safety",
        "equipment_tag": None,
        "severity": "high",
        "date": "2024-03-01",
    },
    {
        "lesson_id": "LL-013",
        "title": "Pump Impeller Erosion from Process Contamination",
        "description": "Pump P-310 impeller showed severe erosion after only 6 months of service. Investigation revealed process fluid contained abrasive particles that were not considered in original material selection.",
        "root_cause": "Process conditions changed after pump was specified. No management of change process was followed for the process modification. Pump metallurgy not reviewed for new operating conditions.",
        "corrective_action": "All process changes must include review of downstream equipment metallurgy. Added pump inspection interval to preventive maintenance based on process conditions. Upgraded impeller material for abrasive service.",
        "category": "process",
        "equipment_tag": "P-310",
        "severity": "medium",
        "date": "2024-04-01",
    },
    {
        "lesson_id": "LL-014",
        "title": "DCS Communication Loss During Maintenance",
        "description": "Distributed Control System (DCS) lost communication to field instruments during routine cabinet maintenance. Caused multiple control loops to fail to manual, requiring operator intervention.",
        "root_cause": "Network switch was inadvertently disconnected during cabinet cleaning. No warning labels on critical network connections. Maintenance procedure did not identify critical components.",
        "corrective_action": "Critical network components labeled 'Do Not Disconnect'. Added network architecture diagram to cabinet door. DCS cabinet maintenance requires control room notification and operator readiness.",
        "category": "instrumentation",
        "equipment_tag": "DCS-001",
        "severity": "high",
        "date": "2024-02-10",
    },
    {
        "lesson_id": "LL-015",
        "title": "Column Tray Damage During Internal Inspection",
        "description": "Distillation column COL-100 trays were damaged when inspection team walked on them without proper support. Trays were not designed to support point loads from personnel.",
        "root_cause": "Entry procedure did not specify tray load limitations. Inspection team unfamiliar with column internal design. No protective walkways installed.",
        "corrective_action": "Column entry procedures must specify load limitations for internals. Portable walkways required for all column internal inspections. Pre-entry briefing must include equipment structural limitations.",
        "category": "mechanical",
        "equipment_tag": "COL-100",
        "severity": "medium",
        "date": "2024-03-20",
    },
    {
        "lesson_id": "LL-016",
        "title": "Relief Valve Testing Failure",
        "description": "Pressure relief valve PSV-400 failed to lift at set pressure during routine testing. Valve had been in service for 3 years without testing. Process operated without adequate overpressure protection.",
        "root_cause": "Relief valve testing schedule was missed due to administrative error. No automated reminder system for critical safety device testing.",
        "corrective_action": "Implemented automated tracking system for all safety-critical device testing. Relief valve testing cannot be deferred without engineering review. Added redundant notification for upcoming test requirements.",
        "category": "safety",
        "equipment_tag": "PSV-400",
        "severity": "critical",
        "date": "2024-01-15",
    },
    {
        "lesson_id": "LL-017",
        "title": "Bearing Lubrication Contamination",
        "description": "Multiple rotating equipment bearings failed prematurely due to contaminated lubricant. Investigation traced contamination to improper lubricant storage and handling practices.",
        "root_cause": "Lubricant containers were stored outside exposed to weather. Transfer containers were not cleaned between different lubricant types. No contamination testing before use.",
        "corrective_action": "All lubricants must be stored indoors in climate-controlled area. Dedicated transfer equipment for each lubricant type. Particle count testing required before adding lubricant to critical equipment.",
        "category": "mechanical",
        "equipment_tag": None,
        "severity": "high",
        "date": "2024-04-10",
    },
    {
        "lesson_id": "LL-018",
        "title": "Scaffolding Collapse During Maintenance",
        "description": "Scaffolding erected for exchanger maintenance collapsed when heavy equipment was placed on platform. No injuries occurred as area was evacuated for equipment lift.",
        "root_cause": "Scaffold was designed for personnel access only, not equipment storage. Load capacity not communicated to work crew. No load rating signs posted.",
        "corrective_action": "All scaffolds must have load rating signs posted visibly. Scaffold design must consider intended use including any equipment or material storage. Pre-use inspection checklist includes load verification.",
        "category": "safety",
        "equipment_tag": "HX-300",
        "severity": "critical",
        "date": "2024-02-05",
    },
    {
        "lesson_id": "LL-019",
        "title": "Motor Overheating After VFD Installation",
        "description": "Motor M-400 experienced overheating after Variable Frequency Drive (VFD) installation. Motor insulation was damaged due to voltage spikes from VFD switching.",
        "root_cause": "Motor was not rated for VFD operation. No harmonic filter installed between VFD and motor. Motor insulation class insufficient for VFD duty.",
        "corrective_action": "All VFD installations require motor compatibility verification. Harmonic filters required for motors not specifically rated for VFD use. Updated motor specification to include VFD duty requirements.",
        "category": "electrical",
        "equipment_tag": "M-400",
        "severity": "high",
        "date": "2024-03-05",
    },
    {
        "lesson_id": "LL-020",
        "title": "Universal Lesson: Shift Handover Communication",
        "description": "Critical maintenance task was incomplete at shift change. Incoming shift assumed task was complete and returned equipment to service, causing process upset.",
        "root_cause": "Verbal shift handover did not clearly communicate task status. No written handover log for ongoing work. Outgoing shift departed before incoming shift understood equipment status.",
        "corrective_action": "Written shift handover log mandatory for all maintenance work in progress. Face-to-face handover required with joint equipment status verification. No equipment returned to service until positive confirmation of task completion.",
        "category": "safety",
        "equipment_tag": None,
        "severity": "high",
        "date": "2024-04-20",
    },
]

# Sample jobs data
SAMPLE_JOBS = [
    {
        "job_id": "JOB-001",
        "job_title": "Pump P-102 Mechanical Seal Replacement",
        "job_description": "Replace mechanical seal on centrifugal pump P-102 during scheduled turnaround. Pump handles hot hydrocarbon service at 150 psi. Work includes isolation, draining, seal removal and installation, and startup verification.",
        "equipment_tag": "P-102",
        "job_type": "replacement",
        "planned_date": "2024-06-15",
    },
    {
        "job_id": "JOB-002",
        "job_title": "Heat Exchanger HX-300 Tube Bundle Extraction",
        "job_description": "Extract tube bundle from shell-and-tube heat exchanger HX-300 for inspection and cleaning. Bundle weight approximately 5 tons. Requires crane lift and specialized rigging.",
        "equipment_tag": "HX-300",
        "job_type": "inspection",
        "planned_date": "2024-06-16",
    },
    {
        "job_id": "JOB-003",
        "job_title": "Motor M-305 Electrical Maintenance",
        "job_description": "Perform electrical maintenance on motor M-305 including insulation testing, connection inspection, and bearing current measurement. Motor is 500 HP, 4160V.",
        "equipment_tag": "M-305",
        "job_type": "maintenance",
        "planned_date": "2024-06-17",
    },
    {
        "job_id": "JOB-004",
        "job_title": "Control Valve V-460 Overhaul",
        "job_description": "Complete overhaul of control valve V-460 including trim replacement, packing replacement, and actuator calibration. Valve is in high pressure steam service.",
        "equipment_tag": "V-460",
        "job_type": "repair",
        "planned_date": "2024-06-18",
    },
    {
        "job_id": "JOB-005",
        "job_title": "Compressor C-200 Bearing Inspection",
        "job_description": "Inspect and replace bearings on reciprocating compressor C-200. Includes shaft alignment verification and vibration baseline measurement after assembly.",
        "equipment_tag": "C-200",
        "job_type": "inspection",
        "planned_date": "2024-06-19",
    },
    {
        "job_id": "JOB-006",
        "job_title": "Tank T-600 Confined Space Entry",
        "job_description": "Enter storage tank T-600 for internal inspection. Tank previously contained crude oil. Requires confined space entry permit, gas testing, and ventilation.",
        "equipment_tag": "T-600",
        "job_type": "inspection",
        "planned_date": "2024-06-20",
    },
    {
        "job_id": "JOB-007",
        "job_title": "Reactor R-300 Flange Gasket Replacement",
        "job_description": "Replace gaskets on reactor R-300 high pressure flanges during turnaround. Operating conditions: 500°F, 300 psi. Critical gasket selection required.",
        "equipment_tag": "R-300",
        "job_type": "replacement",
        "planned_date": "2024-06-21",
    },
    {
        "job_id": "JOB-008",
        "job_title": "Pump P-210 Coupling Alignment",
        "job_description": "Perform laser alignment on pump P-210 after motor replacement. Includes coupling inspection and replacement if required.",
        "equipment_tag": "P-210",
        "job_type": "maintenance",
        "planned_date": "2024-06-22",
    },
    {
        "job_id": "JOB-009",
        "job_title": "Lube Oil Filter Replacement - Multiple Units",
        "job_description": "Replace lube oil filters on compressors C-100, C-200, and C-300. Ensure proper filter orientation and oil system cleanliness.",
        "equipment_tag": None,
        "job_type": "replacement",
        "planned_date": "2024-06-23",
    },
    {
        "job_id": "JOB-010",
        "job_title": "Relief Valve PSV-500 Testing",
        "job_description": "Perform bench testing on pressure relief valve PSV-500. Verify set pressure and reseat pressure. Refurbish or replace as required.",
        "equipment_tag": "PSV-500",
        "job_type": "inspection",
        "planned_date": "2024-06-24",
    },
    {
        "job_id": "JOB-011",
        "job_title": "DCS Cabinet Maintenance",
        "job_description": "Perform preventive maintenance on DCS cabinets including cleaning, connection inspection, and backup verification. Coordinate with operations for any communication interruptions.",
        "equipment_tag": "DCS-002",
        "job_type": "maintenance",
        "planned_date": "2024-06-25",
    },
    {
        "job_id": "JOB-012",
        "job_title": "Column COL-200 Internal Inspection",
        "job_description": "Perform internal inspection of distillation column COL-200. Inspect trays, downcomers, and vessel wall. Requires confined space entry and scaffolding inside column.",
        "equipment_tag": "COL-200",
        "job_type": "inspection",
        "planned_date": "2024-06-26",
    },
    {
        "job_id": "JOB-013",
        "job_title": "VFD Installation for Motor M-500",
        "job_description": "Install new Variable Frequency Drive for motor M-500. Includes motor compatibility verification, cable installation, and commissioning.",
        "equipment_tag": "M-500",
        "job_type": "replacement",
        "planned_date": "2024-06-27",
    },
    {
        "job_id": "JOB-014",
        "job_title": "Pressure Transmitter Calibration",
        "job_description": "Calibrate multiple pressure transmitters PT-200 through PT-210. Verify instrument accuracy and adjust as required.",
        "equipment_tag": "PT-200",
        "job_type": "maintenance",
        "planned_date": "2024-06-28",
    },
    {
        "job_id": "JOB-015",
        "job_title": "Scaffolding Erection for HX-400 Maintenance",
        "job_description": "Erect scaffolding for heat exchanger HX-400 maintenance access. Scaffold must support personnel and light equipment for shell cleaning.",
        "equipment_tag": "HX-400",
        "job_type": "maintenance",
        "planned_date": "2024-06-29",
    },
]


def create_sample_lessons_excel():
    """Create sample lessons learned Excel file."""
    df = pd.DataFrame(SAMPLE_LESSONS)

    # Convert date strings to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Save to Excel
    output_path = "data/sample_lessons.xlsx"
    df.to_excel(output_path, index=False, sheet_name="Lessons Learned")
    print(f"Created {output_path} with {len(df)} lessons")


def create_sample_jobs_excel():
    """Create sample jobs Excel file."""
    df = pd.DataFrame(SAMPLE_JOBS)

    # Convert date strings to datetime
    df["planned_date"] = pd.to_datetime(df["planned_date"])

    # Save to Excel
    output_path = "data/sample_jobs.xlsx"
    df.to_excel(output_path, index=False, sheet_name="Job Descriptions")
    print(f"Created {output_path} with {len(df)} jobs")


if __name__ == "__main__":
    create_sample_lessons_excel()
    create_sample_jobs_excel()
    print("Sample data files created successfully!")
