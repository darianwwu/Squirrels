# Squirrels in Town — Detecting and Tracking the Synurbization of Squirrels in Berlin

**Studyproject:** Squirrels in Town – Detecting and Tracking the Synurbization of Squirrels in Berlin  
**Supervisors:** Benjamin Risse, Luis Garcia-Rodriguez & Sinah Drenske  
**Semester:** WiSe 2025/26

**Group 3:** Jan Becker, Deekshita Ram, Darian Weiß

---

## About this Repository

This repository contains the code and workflows developed as part of the study project *Squirrels in Town*, which investigates the synurbization of squirrels in Berlin. The goal is to automatically analyze camera trap footage and RFID sensor data collected at squirrel feeding boxes across the city.

The repository is organized into **workflow-specific subfolders**, each containing its own notebook, data folders, and documentation.

---

## Repository Structure

```
Squirrels/
├── RFID workflow/          # Synchronize RFID data with box camera footage
│   ├── input/              # Video files + RFID Excel sheet
│   ├── models/             # YOLO weights
│   ├── outputs/            # Auto-generated results
│   ├── squirrel_rfid_workflow.ipynb
│   └── README.md           # Workflow-specific documentation
│
├── ...                     # Further workflow folders will be added here
│
└── README.md               # This file
```

---

## Workflows

### [RFID workflow](RFID%20workflow/README.md)

Synchronizes RFID sensor events with box camera videos to compute:

- Time spent in the box per squirrel
- Number of nuts taken per squirrel (via YOLO object detection)
- A reviewable overlay video of relevant segments
- CSV summary tables for downstream analysis

→ See [RFID workflow/README.md](RFID%20workflow/README.md) for setup and usage instructions.
