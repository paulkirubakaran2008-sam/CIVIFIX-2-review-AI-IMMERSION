"""
Review 2 test script - checks the hybrid engine on (a) the held-out test split and
(b) 24 hand-written complaints with wording that is NOT in dataset.csv.
Run:  python tests.py        (writes outputs/hybrid_report.txt)
"""
from sklearn.metrics import classification_report
from engine import analyse, load_models
from preprocess import load_data, split_data_full
import train_category, train_severity

# (complaint, expected category, expected minimum priority)  - new wording written by hand
UNSEEN = [
    ("The ceiling fan in room 204 is sparking and smells like burning", "Electrical", "P1"),
    ("A live cable is hanging loose from the pole beside the canteen", "Electrical", "P1"),
    ("Tube light in the library is flickering", "Electrical", "P4"),
    ("Main water pipe burst and the ground floor is flooded", "Plumbing", "P1"),
    ("The washbasin tap keeps dripping in the staff room", "Plumbing", "P4"),
    ("Water is seeping through the roof of the seminar hall", "Water Leakage", "P3"),
    ("Sewage is overflowing outside the boys hostel toilets", "Toilet/Sanitation", "P1"),
    ("The girls washroom has no cleaning for two days", "Toilet/Sanitation", "P3"),
    ("Garbage has not been collected near the canteen for a week", "Garbage", "P3"),
    ("Overflowing dustbin beside the library gate", "Garbage", "P4"),
    ("Big pothole on the road near the main gate, bikes are skidding", "Pothole", "P2"),
    ("Small crack on the road near the parking", "Pothole", "P4"),
    ("The walkway to the lab block is broken and students may fall", "Road/Pathway", "P2"),
    ("A wall in the old block has a deep crack and may collapse", "Civil", "P1"),
    ("Paint is peeling off the corridor wall", "Civil", "P4"),
    ("The classroom door hinge is broken and the door is about to fall", "Carpentry", "P1"),
    ("A bench in the seminar hall has a loose leg", "Carpentry", "P4"),
    ("The projector in class 3B is not turning on", "Class Items", "P3"),
    ("Whiteboard marker is missing in the classroom", "Class Items", "P4"),
    ("Computer lab mouse is not working", "Class Items", "P4"),
    ("Emergency exit in the auditorium is blocked by broken chairs", "Carpentry", "P1"),
    ("Water leaking near the electrical panel in the lab", "Water Leakage", "P2"),
    ("Mosquito breeding in stagnant water behind the hostel", "Water Leakage", "P3"),
    ("Fire smell from the switch board in the workshop", "Electrical", "P1"),
]
ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


def main():
    cat_m, sev_m = load_models()          # final models (trained on all data) - used for part B
    out = []

    # (a) hybrid on the held-out split. Part A uses models trained on the TRAIN split only,
    # otherwise the test rows would already have been seen (the saved models use all data).
    df = load_data()
    train, test = split_data_full(df)
    cat_tr = train_category.build_model().fit(train["clean"], train["category"])
    sev_tr = train_severity.build_model().fit(train["clean"], train["severity"])
    res = [analyse(t, cat_tr, sev_tr) for t in test["clean"]]
    ml_pred = [r["ml_severity"] for r in res]
    hy_pred = [r["severity"] for r in res]
    out.append("== A. Held-out test set (83 complaints) - severity ==")
    out.append("ML only:\n" + classification_report(test["severity"], ml_pred, labels=["Low", "Medium", "High"], zero_division=0))
    out.append("Hybrid (ML + safety rules):\n" + classification_report(test["severity"], hy_pred, labels=["Low", "Medium", "High"], zero_division=0))
    rule_hits = sum(r["escalated_by_rules"] for r in res)
    flagged = sum(r["needs_human_review"] for r in res)
    out.append(f"Safety rules raised the severity of {rule_hits} of 83 test complaints; {flagged} flagged for human review.\n")

    # (b) unseen wording
    out.append("== B. Unseen hand-written complaints ==")
    cat_ok = prio_ok = safe_ok = 0
    for text, exp_cat, exp_prio in UNSEEN:
        r = analyse(text, cat_m, sev_m)
        c = r["category"] == exp_cat
        p = ORDER[r["priority"]] <= ORDER[exp_prio]          # at least as urgent as expected
        cat_ok += c
        prio_ok += p
        if exp_prio == "P1":
            safe_ok += r["priority"] == "P1"
        out.append(f"{'OK ' if c else 'XX '}{'OK ' if p else 'XX '}{r['priority']} {r['severity']:<6} {r['category']:<18} <- {text}"
                   + ("  [REVIEW]" if r["needs_human_review"] else ""))
    n = len(UNSEEN)
    n_p1 = sum(1 for _, _, p in UNSEEN if p == "P1")
    out.append(f"\nCategory correct       : {cat_ok}/{n} ({cat_ok/n:.0%})")
    out.append(f"Priority >= expected   : {prio_ok}/{n} ({prio_ok/n:.0%})")
    out.append(f"Critical (P1) caught   : {safe_ok}/{n_p1}")
    text = "\n".join(out)
    print(text)
    open("outputs/hybrid_report.txt", "w").write(text)


if __name__ == "__main__":
    main()
