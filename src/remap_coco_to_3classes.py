"""Remap COCO category names/ids into the final 3-class project taxonomy.
Final IDs: 1 healthy_leaf, 2 unhealthy_leaf, 3 stem.
Edit NAME_MAP if your annotation tool used slightly different class names.
"""
import argparse
import json
import os

NAME_MAP = {
    "healthy_leaf": 1,
    "healthy leaf": 1,
    "leaf_healthy": 1,
    "unhealthy_leaf": 2,
    "unhealthy leaf": 2,
    "diseased_leaf": 2,
    "diseased leaf": 2,
    "leaf_disease_spot": 2,
    "stem_lesion": 2,
    "wilted_leaf": 2,
    "disease_center": 2,
    "stem": 3,
    "stem_branch": 3,
    "branch": 3,
}
CATEGORIES = [
    {"id": 1, "name": "healthy_leaf"},
    {"id": 2, "name": "unhealthy_leaf"},
    {"id": 3, "name": "stem"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.load(open(args.input))
    id_to_name = {c["id"]: c["name"].strip().lower() for c in data.get("categories", [])}
    new_annotations = []
    dropped = 0
    for ann in data.get("annotations", []):
        name = id_to_name.get(ann["category_id"], "")
        new_id = NAME_MAP.get(name)
        if new_id is None:
            dropped += 1
            continue
        ann = dict(ann)
        ann["category_id"] = new_id
        new_annotations.append(ann)

    data["annotations"] = new_annotations
    data["categories"] = CATEGORIES
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(data, open(args.output, "w"), indent=2)
    print(f"Saved {args.output}")
    print(f"Annotations kept: {len(new_annotations)}, dropped: {dropped}")


if __name__ == "__main__":
    main()
