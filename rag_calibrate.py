"""rag_calibrate.py - measure the retrieval distance threshold instead of guessing it.

RAG_DISTANCE_THRESHOLD decides which retrieved rounds the model is allowed to see.
Set too high it feeds in matches from the wrong continent; set too low it starves
the prompt of the one signal that is grounded in the bot's own history.

This reads the embeddings already stored in the index and compares every round
against every other one, so it needs no model inference and takes seconds.

    python rag_calibrate.py
    python rag_calibrate.py --collection geoguessr_carmeta
    python rag_calibrate.py --k 3          # judge by the top-3 vote, as the bot does

The index is built with cosine distance, so 0 means identical and 2 opposite.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

config.QUIET = True
import rag


def norm(c: str) -> str:
    return (c or "").strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="geoguessr_rondas")
    ap.add_argument("--k", type=int, default=1, help="how many neighbours vote (1 = nearest)")
    ap.add_argument("--db", default="./vetores_db")
    ap.add_argument("--ids-from", help="only evaluate rounds also present in this "
                                       "collection, so two encoders are compared "
                                       "on exactly the same images")
    args = ap.parse_args()

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    try:
        col = client.get_collection(args.collection)
    except Exception as e:
        print(f"Could not open collection {args.collection!r}: {e}")
        return 1

    got = col.get(include=["embeddings", "metadatas"])
    ids = got["ids"]
    keep = range(len(ids))
    if args.ids_from:
        other = set(client.get_collection(args.ids_from).get(include=[])["ids"])
        keep = [i for i, _id in enumerate(ids) if _id in other]
        print(f"restricted to {len(keep)} rounds shared with {args.ids_from}")
    emb = np.asarray([got["embeddings"][i] for i in keep], dtype=np.float32)
    countries = [norm(got["metadatas"][i].get("country", "")) for i in keep]
    n = len(emb)
    if n < 50:
        print(f"Only {n} rounds indexed — too few to calibrate.")
        return 1
    print(f"{args.collection}: {n} rounds, {len(set(countries))} distinct countries")
    top = Counter(countries).most_common(5)
    print("  most common: " + ", ".join(f"{c} {k}" for c, k in top))

    # Cosine distance, the space the index was built with.
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)      # a round must never match itself
    dist = 1.0 - sim

    order = np.argsort(dist, axis=1)[:, :max(args.k, 1)]
    rows = []                            # (distance to nearest kept, predicted country, true country)
    for i in range(n):
        idxs = order[i]
        rows.append((float(dist[i, idxs[0]]), [(float(dist[i, j]), countries[j]) for j in idxs],
                     countries[i]))

    same = np.array([r[0] for r in rows if r[1][0][1] == r[2]])
    diff = np.array([r[0] for r in rows if r[1][0][1] != r[2]])
    print(f"\nnearest-neighbour distance, same country   (n={len(same)}): "
          f"median {np.median(same):.3f}  p90 {np.percentile(same, 90):.3f}  max {same.max():.3f}")
    print(f"nearest-neighbour distance, different country (n={len(diff)}): "
          f"median {np.median(diff):.3f}  p10 {np.percentile(diff, 10):.3f}  min {diff.min():.3f}")

    print(f"\n{'threshold':>10}{'shown':>8}{'coverage':>10}{'precision':>11}{'useful':>9}{'misleading':>12}")
    best = None
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80, 1.00]:
        kept = [r for r in rows if r[1][0][0] <= t]
        if not kept:
            continue
        if args.k > 1:
            correct = 0
            for _, cand, truth in kept:
                votes = Counter(c for d, c in cand if d <= t)
                correct += bool(votes) and votes.most_common(1)[0][0] == truth
        else:
            correct = sum(1 for _, cand, truth in kept if cand[0][1] == truth)
        coverage = len(kept) / n
        precision = correct / len(kept)
        useful = correct / n                    # right hint shown, per round played
        misleading = (len(kept) - correct) / n   # wrong hint shown, per round played
        score = useful - misleading
        flag = ""
        if best is None or score > best[1]:
            best, flag = (t, score), ""
        print(f"{t:>10.2f}{len(kept):>8}{coverage:>9.1%}{precision:>11.1%}"
              f"{useful:>9.1%}{misleading:>12.1%}{flag}")

    print(f"\nCurrent RAG_DISTANCE_THRESHOLD = {rag.RAG_DISTANCE_THRESHOLD}")
    print(f"Best separation (useful minus misleading) at {best[0]:.2f}.")
    print("Set it with RAG_DISTANCE_THRESHOLD in .env, then re-run "
          "`python benchmark.py` to confirm it helps end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
