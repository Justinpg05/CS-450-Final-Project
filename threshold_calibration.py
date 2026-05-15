"""
threshold_calibration.py
------------------------
Automatically discovers pairs from:
  ImagePairs/PositivePairs/1-10   (same identity)
  ImagePairs/NegativePairs/1-10   (different identity)

Each numbered folder must contain exactly 2 images.

Run:
  python threshold_calibration.py
"""

import sys
from pathlib import Path

import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_tools.load_known_embeddings import get_embedding  # noqa: E402

BASE       = Path(__file__).resolve().parent
PAIRS_ROOT = BASE / "ImagePairs"
POS_ROOT   = PAIRS_ROOT / "PositivePairs"
NEG_ROOT   = PAIRS_ROOT / "NegativePairs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """Return sorted list of (img_a, img_b, folder_name) from numbered subfolders."""
    pairs = []
    for folder in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not folder.is_dir():
            continue
        imgs = sorted([f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        if len(imgs) < 2:
            print(f"  ⚠  {folder.name}: fewer than 2 images — skipping")
            continue
        pairs.append((imgs[0], imgs[1], folder.name))
    return pairs


def compute_similarity(path_a: Path, path_b: Path) -> float | None:
    emb_a = get_embedding(str(path_a))
    emb_b = get_embedding(str(path_b))
    if emb_a is None or emb_b is None:
        return None
    return F.cosine_similarity(emb_a, emb_b).item()


def find_best_threshold(pos_scores: list[float], neg_scores: list[float]):
    """Sweep candidate thresholds and return the one that maximises accuracy."""
    all_scores = sorted(set(pos_scores + neg_scores))
    candidates = []
    for i, s in enumerate(all_scores):
        candidates.append(s)
        if i < len(all_scores) - 1:
            candidates.append((s + all_scores[i + 1]) / 2)

    best_t, best_acc, best_m = 0.5, -1.0, {}

    for t in sorted(set(candidates)):
        tp = sum(1 for s in pos_scores if s >= t)
        fn = sum(1 for s in pos_scores if s <  t)
        tn = sum(1 for s in neg_scores if s <  t)
        fp = sum(1 for s in neg_scores if s >= t)

        total    = tp + fn + tn + fp
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0)
        far = fp / len(neg_scores) if neg_scores else 0
        frr = fn / len(pos_scores) if pos_scores else 0

        if accuracy > best_acc or (accuracy == best_acc and t > best_t):
            best_acc, best_t = accuracy, t
            best_m = dict(tp=tp, fn=fn, tn=tn, fp=fp,
                          accuracy=accuracy, precision=precision,
                          recall=recall, f1=f1, far=far, frr=frr)

    return best_t, best_m


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 62)
    print("  FaceCandy — Cosine Similarity Threshold Calibration")
    print("=" * 62)

    pos_pairs = load_pairs(POS_ROOT)
    neg_pairs = load_pairs(NEG_ROOT)

    pos_scores: list[float] = []
    neg_scores: list[float] = []

    # ── Positive pairs ────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  POSITIVE PAIRS  (same identity — expect HIGH similarity)")
    print(f"{'─'*62}")

    for img_a, img_b, folder in pos_pairs:
        score = compute_similarity(img_a, img_b)
        label = f"  [Pair {folder:>2}]  {img_a.name}  ↔  {img_b.name}"
        if score is None:
            print(f"{label}\n            ⚠  face not detected — skipping")
        else:
            flag = "✓" if score >= 0.5 else "✗ (unexpectedly low)"
            print(f"{label}\n            similarity: {score:.4f}  {flag}")
            pos_scores.append(score)

    # ── Negative pairs ────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  NEGATIVE PAIRS  (different identity — expect LOW similarity)")
    print(f"{'─'*62}")

    for img_a, img_b, folder in neg_pairs:
        score = compute_similarity(img_a, img_b)
        label = f"  [Pair {folder:>2}]  {img_a.name}  ↔  {img_b.name}"
        if score is None:
            print(f"{label}\n            ⚠  face not detected — skipping")
        else:
            flag = "✓" if score < 0.8 else "✗ (unexpectedly high)"
            print(f"{label}\n            similarity: {score:.4f}  {flag}")
            neg_scores.append(score)

    # ── Score summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  SCORE SUMMARY")
    print(f"{'─'*62}")
    if pos_scores:
        print(f"  Positive — mean: {sum(pos_scores)/len(pos_scores):.4f}"
              f"   min: {min(pos_scores):.4f}   max: {max(pos_scores):.4f}")
    if neg_scores:
        print(f"  Negative — mean: {sum(neg_scores)/len(neg_scores):.4f}"
              f"   min: {min(neg_scores):.4f}   max: {max(neg_scores):.4f}")

    if not pos_scores or not neg_scores:
        print("\n  ⚠  Not enough valid pairs to compute a threshold.")
        return

    # ── Threshold search ──────────────────────────────────────────────────────
    threshold, m = find_best_threshold(pos_scores, neg_scores)

    overlap = max(neg_scores) > min(pos_scores)

    print(f"\n{'='*62}")
    print(f"  RECOMMENDED THRESHOLD:  {threshold:.4f}")
    print(f"{'='*62}")
    print(f"  Accuracy  : {m['accuracy']:.1%}  "
          f"({m['tp']+m['tn']}/{m['tp']+m['fn']+m['tn']+m['fp']} pairs correct)")
    print(f"  Precision : {m['precision']:.1%}")
    print(f"  Recall    : {m['recall']:.1%}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  FAR (impostors accepted) : {m['far']:.1%}")
    print(f"  FRR (real people denied) : {m['frr']:.1%}")
    print(f"\n  TP={m['tp']}  FN={m['fn']}  TN={m['tn']}  FP={m['fp']}")

    if overlap:
        print(f"\n  ⚠  Score ranges overlap — perfect separation not possible")
        print(f"     with these pairs. Fine-tuning will improve this.")

    print(f"\n  → Set THRESHOLD = {threshold:.2f} in agent_tools/config.py\n")


if __name__ == "__main__":
    run()
