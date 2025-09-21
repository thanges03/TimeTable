import argparse
import pandas as pd
import math
from ortools.sat.python import cp_model
from collections import defaultdict

DAYS = 5
PERIODS = 8
SLOTS = DAYS * PERIODS

def find_sheet(xl, candidates):
    low = [s.lower() for s in xl.sheet_names]
    for c in candidates:
        cl = c.lower()
        for i, sname in enumerate(low):
            if cl in sname:
                return xl.sheet_names[i]
    return None

def expand_tasks_for_class(subjects_df, class_row):
    cid = class_row["class_id"]
    strength = int(class_row.get("strength", 60))
    rows = subjects_df[subjects_df["class_id"] == cid]
    tasks = []
    for _, r in rows.iterrows():
        typ = str(r.get("type", "theory")).strip().lower()
        sid = r["subject_id"]
        sname = r.get("subject_name", sid)
        staff = r.get("main_staff_id", None) or r.get("staff_id", "TBD")
        hours = int(r.get("hours_per_week", 0) or 0)
        if typ == "theory":
            for h in range(hours):
                tasks.append({
                    "task_id": f"{sid}_T_{h}",
                    "subject_id": sid,
                    "subject_name": sname,
                    "duration": 1,
                    "kind": "theory",
                    "staff": staff
                })
        elif typ == "lab":
            sessions = math.ceil(hours / 2)
            batches = math.ceil(strength / 30)
            for b in range(1, batches + 1):
                for s in range(sessions):
                    tasks.append({
                        "task_id": f"{sid}_L_B{b}_S{s}",
                        "subject_id": sid,
                        "subject_name": f"{sname} (B{b})",
                        "duration": 2,
                        "kind": "lab",
                        "staff": staff
                    })
        else:
            # fallback treat as theory
            for h in range(hours):
                tasks.append({
                    "task_id": f"{sid}_T_{h}",
                    "subject_id": sid,
                    "subject_name": sname,
                    "duration": 1,
                    "kind": "theory",
                    "staff": staff
                })
    return tasks

def solve_class(tasks, staff_daily_limit=5, enforce_one_lab_per_day=True, time_limit=60,w_adj=12, w_double_day=6, w_gaps=8, w_over2=10):
    if not tasks:
        return None, None, "NO_TASKS"
    model = cp_model.CpModel()
    T = SLOTS
    for t in tasks:
        dur = t["duration"]
        t["start"] = model.NewIntVar(0, T - dur, f"start_{t['task_id']}")
        t["end"]   = model.NewIntVar(0, T, f"end_{t['task_id']}")
        t["interval"] = model.NewIntervalVar(t["start"], dur, t["end"], f"int_{t['task_id']}")
        if dur == 2:
            for d in range(DAYS):
                forbidden_start = d*PERIODS + (PERIODS - 1)
                model.Add(t["start"] != forbidden_start)
    model.AddNoOverlap([t["interval"] for t in tasks])
    staff_to_intervals = defaultdict(list)
    for t in tasks:
        staff_to_intervals[t["staff"]].append(t["interval"])
    for staff, ints in staff_to_intervals.items():
        if len(ints) > 1:
            model.AddNoOverlap(ints)
    if enforce_one_lab_per_day:
        for d in range(DAYS):
            lab_flags = []
            for idx, t in enumerate(tasks):
                if t["duration"] == 2:
                    b = model.NewBoolVar(f"is_{t['task_id']}_day{d}")
                    # if b then start in day d
                    model.Add(t["start"] >= d*PERIODS).OnlyEnforceIf(b)
                    model.Add(t["start"] < (d+1)*PERIODS).OnlyEnforceIf(b)
                    model.Add(t["start"] < (d+1)*PERIODS).OnlyEnforceIf(b)
                    lab_flags.append(b)
            if lab_flags:
                model.Add(sum(lab_flags) <= 1)
    occ = {}
    for i, t in enumerate(tasks):
        dur = t["duration"]
        possible_starts = range(0, T - dur + 1)
        is_start = {}
        for v in possible_starts:
            b = model.NewBoolVar(f"isstart_{i}_{v}")
            is_start[v] = b
            model.Add(t["start"] == v).OnlyEnforceIf(b)
            model.Add(t["start"] != v).OnlyEnforceIf(b.Not())
        model.Add(sum(is_start[v] for v in possible_starts) == 1)
        for slot in range(T):
            bslot = model.NewBoolVar(f"occ_{i}_{slot}")
            occ[(i, slot)] = bslot
            covering_starts = [v for v in possible_starts if v <= slot < v + dur]
            if covering_starts:
                model.Add(bslot == sum(is_start[v] for v in covering_starts))
            else:
                model.Add(bslot == 0)

    # Each slot has at most 1 task (class no-overlap enforced by intervals, but keep consistency)
    for slot in range(T):
        model.Add(sum(occ[(i, slot)] for i in range(len(tasks))) <= 1)

    # Staff daily load
    for staff, ints in staff_to_intervals.items():
        for d in range(DAYS):
            slots_day = range(d*PERIODS, (d+1)*PERIODS)
            staff_occ = []
            for i, t in enumerate(tasks):
                if t["staff"] != staff:
                    continue
                for slot in slots_day:
                    staff_occ.append(occ[(i, slot)])
            if staff_occ:
                model.Add(sum(staff_occ) <= staff_daily_limit)

    # Build subj_slot[(subject,slot)] boolean channeling with occ's
    subjects_set = sorted(set(t["subject_id"] for t in tasks))
    subj_slot = {}
    subject_task_indices = defaultdict(list)
    for i, t in enumerate(tasks):
        subject_task_indices[t["subject_id"]].append(i)
    for subj in subjects_set:
        for slot in range(T):
            b = model.NewBoolVar(f"subjslot_{subj}_{slot}")
            subj_slot[(subj, slot)] = b
            # sum occ[i,slot] for i in indices == b
            inds = subject_task_indices[subj]
            if inds:
                model.Add(sum(occ[(i, slot)] for i in inds) == b)
            else:
                model.Add(b == 0)

    # Soft constraints:
    adj_bools = []        # adjacent same-subject in same day
    double_day_flags = [] # subject has >=2 occurrences on a day
    over2_ints = []       # occurrences beyond 2 per day (integer)
    gap_bools = []        # pattern occupied-empty-occupied

    # Adjacent penalty and gap detection and per-day counts
    for subj in subjects_set:
        for d in range(DAYS):
            # count occurrences this day for subject
            slots_day = [d*PERIODS + p for p in range(PERIODS)]
            sum_slots = sum(subj_slot[(subj, s)] for s in slots_day)
            # double day indicator
            dbl = model.NewBoolVar(f"double_{subj}_day{d}")
            double_day_flags.append(dbl)
            # If sum_slots >= 2 => dbl=1; if sum_slots <=1 => dbl=0
            model.Add(sum_slots >= 2).OnlyEnforceIf(dbl)
            model.Add(sum_slots <= 1).OnlyEnforceIf(dbl.Not())

            # over2 integer: max(0, sum_slots - 2)
            over2 = model.NewIntVar(0, PERIODS, f"over2_{subj}_day{d}")
            over2_ints.append(over2)
            # over2 >= sum_slots - 2
            model.Add(over2 >= sum_slots - 2)
            # over2 <= sum_slots (trivial)
            model.Add(over2 <= sum_slots)

            # adjacent
            for p in range(PERIODS - 1):
                s1 = d*PERIODS + p
                s2 = d*PERIODS + p + 1
                vb = model.NewBoolVar(f"adj_{subj}_{d}_{p}")
                # vb == 1 iff subj_slot(s1) == 1 and subj_slot(s2) == 1
                model.Add(vb <= subj_slot[(subj, s1)])
                model.Add(vb <= subj_slot[(subj, s2)])
                model.Add(vb >= subj_slot[(subj, s1)] + subj_slot[(subj, s2)] - 1)
                adj_bools.append(vb)

    # Gaps penalty: occupied - empty - occupied pattern in a day
    for d in range(DAYS):
        for p in range(PERIODS - 2):
            slot1 = d*PERIODS + p
            slot2 = d*PERIODS + p + 1
            slot3 = d*PERIODS + p + 2
            # occ_any_slot = sum over subj_slot of subj occupying that slot (since at most one subject per slot)
            # We can build occ_any slot by summing subj_slot over subjects
            occ1 = model.NewBoolVar(f"occ_any_{slot1}")
            occ2 = model.NewBoolVar(f"occ_any_{slot2}")
            occ3 = model.NewBoolVar(f"occ_any_{slot3}")
            # occ1 == sum subj_slot(subj,slot1) (0/1)
            model.Add(sum(subj_slot[(s, slot1)] for s in subjects_set) == occ1)
            model.Add(sum(subj_slot[(s, slot2)] for s in subjects_set) == occ2)
            model.Add(sum(subj_slot[(s, slot3)] for s in subjects_set) == occ3)
            # gap bool: occ1==1 and occ2==0 and occ3==1
            gapb = model.NewBoolVar(f"gap_{d}_{p}")
            model.Add(gapb <= occ1)
            model.Add(gapb <= occ3)
            model.Add(gapb <= (1 - occ2))
            # ensure gapb implies occ1=1 and occ3=1 and occ2=0 via implications
            model.Add(occ1 == 1).OnlyEnforceIf(gapb)
            model.Add(occ3 == 1).OnlyEnforceIf(gapb)
            model.Add(occ2 == 0).OnlyEnforceIf(gapb)
            gap_bools.append(gapb)

    # Objective: weighted combination
    adj_sum = sum(adj_bools)
    double_day_sum = sum(double_day_flags)
    over2_sum = sum(over2_ints)
    gap_sum = sum(gap_bools)

    model.Minimize(w_adj * adj_sum + w_double_day * double_day_sum + w_gaps * gap_sum + w_over2 * over2_sum)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, solver.StatusName(status)

    # Build schedule list
    schedule = [""] * T
    for i, t in enumerate(tasks):
        s_val = solver.Value(t["start"])
        for k in range(t["duration"]):
            idx = s_val + k
            if 0 <= idx < T:
                schedule[idx] = t["subject_id"]

    penalty_info = {
        "adjacent": int(solver.Value(adj_sum)),
        "double_days": int(solver.Value(double_day_sum)),
        "over2": int(solver.Value(over2_sum)),
        "gaps": int(solver.Value(gap_sum)),
        "objective": solver.ObjectiveValue()
    }
    return schedule, penalty_info, "FEASIBLE"

# ---------------------------
# Run all classes and export
# ---------------------------
def run_all(input_file, output_file, time_limit_per_class=60, staff_daily_limit=5, no_one_lab_day=False):
    xl = pd.ExcelFile(input_file)
    subjects_sheet = find_sheet(xl, ["Subjects_Final"])#, "subjects", "subjects_final", "subjects_prepared"
    classes_sheet  = find_sheet(xl, ["Classes_Cleaned"])#, "classes", "class_list", "classes_final"
    staffs_sheet   = find_sheet(xl, ["Staffs_Cleaned"])#, "staffs", "faculty"

    if not subjects_sheet or not classes_sheet or not staffs_sheet:
        raise ValueError(f"Required sheets not found in {input_file}. Available: {xl.sheet_names}")

    subjects = pd.read_excel(xl, sheet_name=subjects_sheet)
    classes  = pd.read_excel(xl, sheet_name=classes_sheet)
    staffs   = pd.read_excel(xl, sheet_name=staffs_sheet)

    # normalize column names
    subjects.columns = [c.strip() for c in subjects.columns]
    classes.columns = [c.strip() for c in classes.columns]
    staffs.columns = [c.strip() for c in staffs.columns]

    writer = pd.ExcelWriter(output_file, engine="openpyxl")
    diagnostics = []

    for _, cl in classes.iterrows():
        cid = cl["class_id"]
        print("[INFO] scheduling:", cid)
        tasks = expand_tasks_for_class(subjects, cl)
        if not tasks:
            pd.DataFrame([{"Message":"No subjects"}]).to_excel(writer, sheet_name=f"{cid}_TT", index=False)
            diagnostics.append({"class_id": cid, "status": "NO_SUBJECTS", "penalty": None})
            continue

        schedule, penalty_info, status = solve_class(tasks, staff_daily_limit=staff_daily_limit,
                                                     enforce_one_lab_per_day=(not no_one_lab_day),
                                                     time_limit=time_limit_per_class)
        if schedule is None:
            pd.DataFrame([{"Message":f"No feasible timetable (status={status})"}]).to_excel(writer, sheet_name=f"{cid}_TT", index=False)
            diagnostics.append({"class_id": cid, "status": status, "penalty": None})
            continue

        # Convert schedule to grid and fill "FREE"
        grid = []
        for d in range(DAYS):
            row = []
            for p in range(PERIODS):
                subj = schedule[d*PERIODS + p]
                row.append(subj if subj else "FREE")
            grid.append(row)
        df = pd.DataFrame(grid, index=[f"Day{d+1}" for d in range(DAYS)], columns=[f"P{p+1}" for p in range(PERIODS)])
        df.to_excel(writer, sheet_name=f"{cid}_TT")

        # Lookup sheet: subjects & staff for this class
        subs_for_class = subjects[subjects["class_id"] == cid][["subject_id", "subject_code", "subject_name", "main_staff_id", "supporting_staff_id"]].drop_duplicates()
        subs_for_class = subs_for_class.rename(columns={"main_staff_id":"staff_id"})
        if "name" in staffs.columns:
            subs_for_class = subs_for_class.merge(staffs[["staff_id","name"]].drop_duplicates(), on="staff_id", how="left").rename(columns={"name":"staff_name"})
        else:
            subs_for_class["staff_name"] = subs_for_class["staff_id"]
        subs_for_class.to_excel(writer, sheet_name=f"{cid}_Lookup", index=False)

        diagnostics.append({"class_id": cid, "status": status, "penalty_score": (penalty_info["objective"] if penalty_info else None),
                            "adjacent": penalty_info["adjacent"], "double_days": penalty_info["double_days"],
                            "over2": penalty_info["over2"], "gaps": penalty_info["gaps"]})

    pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)
    writer.close()
    print("[DONE] Written:", output_file)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared dataset Excel file")
    parser.add_argument("--out", default="timetable_output_all.xlsx", help="Output workbook")
    parser.add_argument("--time_limit", type=int, default=120, help="Seconds per class solve")
    parser.add_argument("--staff_daily_limit", type=int, default=5, help="Max periods a staff can teach per day")
    parser.add_argument("--no_one_lab_day", action="store_true", help="Allow more than one lab per class per day")
    args = parser.parse_args()

    run_all(args.input, args.out, time_limit_per_class=args.time_limit, staff_daily_limit=args.staff_daily_limit, no_one_lab_day=args.no_one_lab_day)

