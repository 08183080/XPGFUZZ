import argparse
import json
import os
import random
import sys
from pathlib import Path

# Local import: fine-grained mutator and mutate logic
from .finegrained_mutator import FineGrainedMutator


def load_grammars(grammars_path: Path):
    if not grammars_path.exists():
        raise FileNotFoundError(f"Grammars file not found: {grammars_path}")
    with grammars_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mutate_once(seed_data: str, grammars) -> str:
    mutator = FineGrainedMutator()
    candidates = []

    for grammar in grammars:
        try:
            import re
            regex = re.compile(grammar["regex"], re.DOTALL)

            for match in re.finditer(regex, seed_data):
                captured_values = match.groups()
                placeholders = grammar.get("placeholders", [])
                if len(captured_values) != len(placeholders):
                    continue

                for i, original_value in enumerate(captured_values):
                    ph = placeholders[i]
                    new_values = mutator.mutate(ph["type"], original_value)
                    for new_val in new_values:
                        start, end = match.span(i + 1)
                        mutated_seed = seed_data[:start] + new_val + seed_data[end:]
                        candidates.append(mutated_seed)
        except Exception:
            continue

    if not candidates:
        return ""
    return random.choice(candidates)


def main():
    parser = argparse.ArgumentParser(description="One-shot fine-grained template mutation")
    parser.add_argument("--grammars", required=True, help="Path to grammars.json")
    parser.add_argument("--in", dest="in_path", required=True, help="Input file path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output file path")
    args = parser.parse_args()

    grammars_path = Path(args.grammars)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        seed_data = f.read()

    grammars = load_grammars(grammars_path)
    mutated = mutate_once(seed_data, grammars)

    if not mutated:
        # No mutation produced; exit code 2 means no-op
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(mutated)

    return 0


if __name__ == "__main__":
    sys.exit(main())


