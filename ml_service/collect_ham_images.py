import argparse
import csv
from pathlib import Path
import shutil


def collect(csv_path: Path, search_root: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids = set()
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get('image_id') or '').strip()
            if image_id:
                image_ids.add(image_id)

    count = 0
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for p in search_root.rglob(ext):
            stem = p.stem
            if stem in image_ids:
                dest = out_dir / (stem + p.suffix.lower())
                try:
                    shutil.copy2(p, dest)
                    count += 1
                except Exception:
                    pass
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--search-root', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    copied = collect(Path(args.csv), Path(args.search_root), Path(args.out_dir))
    print(copied)


if __name__ == '__main__':
    main()












