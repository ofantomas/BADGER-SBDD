#!/usr/bin/env python
"""make_pocket.py
Extract atoms within a given radius (Å) from a PDB file around a user-provided center and
write them to a new PDB representing the binding pocket. Hydrogens are excluded by default.

Usage
-----
python make_pocket.py input.pdb cx cy cz radius [output.pdb]

Example
-------
python make_pocket.py 1pxx.pdb 27.116 24.090 14.936 10 1pxx_pocket.pdb
"""

import sys
import math
from typing import List, Tuple

ATOM_PREFIXES = ("ATOM", "HETATM")


def _parse_xyz(line: str) -> Tuple[float, float, float]:
    """Extract x, y, z coordinates from a PDB ATOM/HETATM line."""
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        raise ValueError(f"Failed to parse coordinates from line: {line}")
    return x, y, z


def _is_heavy_atom(line: str) -> bool:
    """Return True if the PDB line represents a non-hydrogen atom."""
    element = line[76:78].strip().upper()
    return element != "H"


def select_pocket(
    lines: List[str], center: Tuple[float, float, float], radius: float
) -> List[str]:
    """Return PDB lines whose atoms lie within *radius* Å of *center*."""
    cx, cy, cz = center
    r2 = radius * radius
    pocket = []
    for ln in lines:
        if not ln.startswith(ATOM_PREFIXES):
            continue
        if not _is_heavy_atom(ln):
            continue
        x, y, z = _parse_xyz(ln)
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
            pocket.append(ln.rstrip("\n"))
    return pocket


def write_pocket(pocket_lines: List[str], out_path: str):
    with open(out_path, "w") as f:
        f.write("HEADER    POCKET\n")
        for ln in pocket_lines:
            f.write(f"{ln}\n")
        f.write("END\n")


def main():
    if len(sys.argv) < 6:
        print(
            "Usage: python make_pocket.py input.pdb cx cy cz radius [output.pdb]",
            file=sys.stderr,
        )
        sys.exit(1)

    input_pdb = sys.argv[1]
    cx, cy, cz = map(float, sys.argv[2:5])
    radius = float(sys.argv[5])
    output_pdb = sys.argv[6] if len(sys.argv) > 6 else "pocket.pdb"

    with open(input_pdb) as fh:
        lines = fh.readlines()

    pocket = select_pocket(lines, (cx, cy, cz), radius)
    write_pocket(pocket, output_pdb)
    print(f"Saved {len(pocket)} atoms to {output_pdb}")


if __name__ == "__main__":
    main()
