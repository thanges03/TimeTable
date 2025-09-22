**Overview**
This project is a timetable generator built using Google OR-Tools (CP-SAT Solver) and Python.
It creates optimal schedules for classes while respecting hard constraints and improving timetable balance with soft constraints.

**The solver ensures:**
No overlap of staff across classes.
No overlap of subjects in a class.
Staff workload per day is within a limit.
Balanced distribution of subjects across the week.
The output is written to Excel, with one timetable per class and a diagnostics sheet.

**Tech Stack**
Python 3.9+
Pandas – Data handling
OpenPyXL – Excel writing
OR-Tools (CP-SAT) – Constraint Programming Solver

**Usage**
Install dependencies:   pip install ortools pandas openpyxl
Run: python timetable_solver_spread_balance.py --input SCE_auto_dataset.xlsx --out timetable_output.xlsx --time_limit 60

**Output**
Excel workbook with:
Class-wise timetable sheets.
Lookup tables for subject–staff mapping.
Diagnostics with penalty scores.
