"""
stage13_combined_output.py
Merges the crossing-intention module's output with the base pipeline's
gender/attribute output (stage7) into one unified per-person record.

Two independent things get computed per tracked pedestrian:
  - gender/hat/backpack/clothing (stage6/7)      -> identity-level, one
    fused value per person, from K diverse still crops
  - crossing intention probability (stage12)     -> WINDOW-level, one
    probability per (pedestrian, K1-second window, K2 offset) triple,
    since intention is inherently time-varying, not a fixed identity trait

So the combined record is keyed by (person_id, window), carrying that
window's crossing probability alongside the person's (constant) fused
attributes.
"""

import csv
import json
import os


COMBINED_FIELDS = [
    "video_id", "person_id", "window_start_sec", "window_end_sec", "K2_sec",
    "crossing_intention_prob", "crossing_predicted",
    "gender", "hat", "backpack", "upper_colour", "upper_style",
    "lower_colour", "lower_style", "attribute_confidence",
]


def build_combined_output(video_id, intention_predictions_path, attribute_summary_path,
                           out_csv="work/output/combined_person_records.csv",
                           out_json="work/output/combined_person_records.json",
                           threshold=0.5):
    """intention_predictions.json: list of
        {"track_id":.., "window":[start,end], "K2":.., "probability":..}
    attribute_summary.json (stage7's per_person_summary.json):
        {"<track_id>": {"gender":{"class":..,"confidence":..}, ...}}
    """
    with open(intention_predictions_path) as f:
        intention_preds = json.load(f)
    with open(attribute_summary_path) as f:
        attr_summary = json.load(f)

    rows = []
    for pred in intention_preds:
        tid = str(pred["track_id"])
        attrs = attr_summary.get(tid, {})

        def _get_val(k):
            v = attrs.get(k)
            if isinstance(v, dict):
                return v.get("class")
            return v

        conf_dict = attrs.get("confidence", {})
        if isinstance(conf_dict, dict) and conf_dict:
            mean_conf = round(sum(conf_dict.values()) / len(conf_dict), 4)
        else:
            mean_conf = None

        rows.append({
            "video_id": video_id,
            "person_id": tid,
            "window_start_sec": pred["window"][0],
            "window_end_sec": pred["window"][1],
            "K2_sec": pred.get("K2"),
            "crossing_intention_prob": round(pred["probability"], 4),
            "crossing_predicted": int(pred["probability"] >= threshold),
            "gender": _get_val("gender"),
            "hat": _get_val("hat"),
            "backpack": _get_val("backpack"),
            "upper_colour": _get_val("upper_colour"),
            "upper_style": _get_val("upper_style"),
            "lower_colour": _get_val("lower_colour"),
            "lower_style": _get_val("lower_style"),
            "attribute_confidence": mean_conf,
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} combined records -> {out_csv} / {out_json}")
    return out_csv, out_json


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", required=True)
    ap.add_argument("--intention_preds", default="work/output/intention_predictions.json")
    ap.add_argument("--attr_summary", default="work/output/per_person_summary.json")
    ap.add_argument("--out_csv", default="work/output/combined_person_records.csv")
    ap.add_argument("--out_json", default="work/output/combined_person_records.json")
    args = ap.parse_args()
    build_combined_output(args.video_id, args.intention_preds, args.attr_summary,
                           args.out_csv, args.out_json)
