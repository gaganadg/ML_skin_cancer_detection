from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter


def make_skin_bg(w=224, h=224):
    # simple skin-like base color with slight noise
    base = Image.new('RGB', (w, h), (220, 190, 170))
    return base


def draw_circle_lesion(img: Image.Image, center=None, radius=None, color=(90, 60, 50)):
    w, h = img.size
    if center is None:
        center = (random.randint(w//3, 2*w//3), random.randint(h//3, 2*h//3))
    if radius is None:
        radius = random.randint(15, 35)
    x, y = center
    draw = ImageDraw.Draw(img)
    bbox = [x-radius, y-radius, x+radius, y+radius]
    draw.ellipse(bbox, fill=color)


def draw_irregular_lesion(img: Image.Image, points=18, jitter=20, color=(60, 35, 30)):
    w, h = img.size
    cx = random.randint(w//3, 2*w//3)
    cy = random.randint(h//3, 2*h//3)
    r = random.randint(18, 40)
    pts = []
    for i in range(points):
        ang = 2*3.14159*i/points
        rr = r + random.randint(-jitter, jitter)
        xx = int(cx + rr * (random.uniform(0.9, 1.1)) * (1.0 if i%2==0 else 0.7) * (0.5 + 0.5*random.random()) * (1.2*random.random()) * (1.0 if random.random()>0.1 else -1.0) * (0.5 + 0.5*random.random()))
        yy = int(cy + rr * (random.uniform(0.9, 1.1)) * (1.0 if i%2==0 else 0.7) * (0.5 + 0.5*random.random()) * (1.2*random.random()) * (1.0 if random.random()>0.1 else -1.0) * (0.5 + 0.5*random.random()))
        pts.append((xx, yy))
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=color)


def generate_split(root: Path, split: str, n_per_class: int = 40):
    (root / split / 'benign').mkdir(parents=True, exist_ok=True)
    (root / split / 'malignant').mkdir(parents=True, exist_ok=True)

    # benign: smoother round lesions
    for i in range(n_per_class):
        img = make_skin_bg()
        draw_circle_lesion(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        img.save(root / split / 'benign' / f'benign_{i:03d}.jpg', quality=92)

    # malignant: irregular darker lesions
    for i in range(n_per_class):
        img = make_skin_bg()
        draw_irregular_lesion(img)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        img.save(root / split / 'malignant' / f'malignant_{i:03d}.jpg', quality=92)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data')
    p.add_argument('--train-n', type=int, default=80)
    p.add_argument('--val-n', type=int, default=20)
    args = p.parse_args()
    root = Path(args.out)
    generate_split(root, 'train', n_per_class=args.train_n)
    generate_split(root, 'val', n_per_class=args.val_n)
    print(f"Synthetic dataset written to {root}")


if __name__ == '__main__':
    main()


