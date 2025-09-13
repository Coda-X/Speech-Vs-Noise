#not in video
import os, csv

MIS_PATH = "features/analysis/day8_misclassified.csv"

def main():
    if not os.path.exists(MIS_PATH):
        print("No misclassified.csv found. Run day8_analyze_errors.py first.")
        return

    rows = []
    with open(MIS_PATH, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print("No mistakes today 🎉 — consider adding harder noises.")
        return


    def wrong_conf(row):
        try:
            p0 = float(row.get("prob_noise") or 0.0)
            p1 = float(row.get("prob_speech") or 0.0)
            pred = int(row["pred_label"])
            return [p0, p1][pred]
        except:
            return 0.0

    rows.sort(key=wrong_conf, reverse=True)
    print("Top confusing cases (most confident wrong):")
    for r in rows[:5]:
        print(f"- {os.path.basename(r['file'])} | true={r['true_label']} pred={r['pred_label']} | p_noise={r['prob_noise']} p_speech={r['prob_speech']}")

if __name__ == "__main__":
    main()
