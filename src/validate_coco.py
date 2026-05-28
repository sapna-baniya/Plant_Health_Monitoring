import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()

    data = json.load(open(args.annotations))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    counts = Counter(a["category_id"] for a in data.get("annotations", []))
    missing = []
    for img in data.get("images", []):
        if not os.path.exists(os.path.join(args.image_dir, img["file_name"])):
            missing.append(img["file_name"])

    print("Images:", len(data.get("images", [])))
    print("Annotations:", len(data.get("annotations", [])))
    print("Categories:", cats)
    print("Counts:")
    for cid, count in sorted(counts.items()):
        print(f"  {cid} {cats.get(cid, 'UNKNOWN')}: {count}")
    print("Missing image files:", len(missing))
    if missing[:10]:
        print("Examples:", missing[:10])


if __name__ == "__main__":
    main()
