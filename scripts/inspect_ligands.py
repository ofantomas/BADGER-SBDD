#!/usr/bin/env python
"""
CLI script converted from the former *inspect_ligands.ipynb* notebook.

The script evaluates generated ligand tensors (``.pt``) by:
1. Reconstructing RDKit molecules from the *pos* / *v* tensors.
2. Converting each ligand to SDF → PDBQT (via Meeko) and docking with AutoDock Vina.
3. Computing chemical metrics (QED, SA, LogP, Lipinski, ring sizes).
4. Checking geometric stability and bond-length distributions.
5. Printing summary statistics to the console and an optional log-file.

Usage
-----
::

    python scripts/inspect_ligands.py \
        --ligands results/ligands/1pxx_10_ligands_large.pt \
        --protein pockets/1pxx_clean.pdbqt \
        --center 27.116 24.090 14.936 \
        --box-size 10 10 10 \
        --tmp-dir results/ligands_sdf/1pxx_10_large \
        --exhaustiveness 128

Notes
-----
*The script heavily relies on the util modules already present in BADGER-SBDD.*
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List
from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import shutil
import csv

sys.path.append(os.getcwd())

from utils import misc, reconstruct, transforms
from utils.evaluation import analyze, eval_atom_type, eval_bond_length, scoring_func
from utils.evaluation.docking_vina import PrepLig, VinaDock


def evaluate_single_ligand(
    ligand_pdbqt: str,
    protein_pdbqt: str,
    center: List[float],
    box_size: List[float],
    mode: str = "dock",
    exhaustiveness: int = 8,
):
    """Dock a single ligand PDBQT with Vina and return affinity & pose."""

    dock = VinaDock(ligand_pdbqt, protein_pdbqt)
    dock.pocket_center = center
    dock.box_size = box_size

    score, pose = dock.dock(
        score_func="vina",
        mode=mode,
        exhaustiveness=exhaustiveness,
        save_pose=True,
    )
    return {"affinity": score, "pose": pose}


def print_dict(d: dict, logger):
    for k, v in d.items():
        logger.info(f"{k}:\t{v:.4f}" if v is not None else f"{k}:\tNone")


def print_ring_ratio(all_ring_sizes: List[Counter], logger):
    for ring_size in range(3, 10):
        n_mol = sum(1 for counter in all_ring_sizes if ring_size in counter)
        logger.info(
            f"ring size: {ring_size} ratio: {n_mol / max(len(all_ring_sizes), 1):.3f}"
        )


def run_evaluation(
    ligands_path: str,
    protein_pdbqt: str,
    center: List[float],
    box_size: List[float],
    result_path: str,
    exhaustiveness: int = 8,
    save_top_k: int = 100,
    save: bool = False,
):
    Path(result_path).mkdir(parents=True, exist_ok=True)
    logger = misc.get_logger("evaluate", log_dir=str(Path(result_path)))

    # ------------------------------------------------------------------
    # Load ligand tensors
    # ------------------------------------------------------------------
    ligands = torch.load(ligands_path, weights_only=False)
    if "pos" not in ligands or "v" not in ligands:
        raise ValueError("Ligands file must contain 'pos' and 'v' keys")

    num_samples = len(ligands["pos"])
    logger.info(f"Loaded {num_samples} ligand samples from {ligands_path}")

    # ------------------------------------------------------------------
    # Statistics accumulators
    # ------------------------------------------------------------------
    all_mol_stable = all_atom_stable = all_n_atom = 0
    n_recon_success = n_eval_success = n_complete = 0

    results = []
    all_pair_dist: List[float] = []
    all_bond_dist: List[float] = []
    all_atom_types: Counter = Counter()
    success_pair_dist: List[float] = []
    success_atom_types: Counter = Counter()

    # ------------------------------------------------------------------
    # Iterate over ligands
    # ------------------------------------------------------------------
    for i in tqdm(range(num_samples), desc="Evaluating ligands"):
        pred_pos, pred_v = ligands["pos"][i], ligands["v"][i]

        pred_atom_type = transforms.get_atomic_number_from_index(
            pred_v, mode="add_aromatic"
        )
        all_atom_types += Counter(pred_atom_type)
        r_stable = analyze.check_stability(pred_pos, pred_atom_type)
        all_mol_stable += r_stable[0]
        all_atom_stable += r_stable[1]
        all_n_atom += r_stable[2]

        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        # --------------------------------------------------------------
        # Reconstruct rdkit molecule
        # --------------------------------------------------------------
        try:
            pred_aromatic = transforms.is_aromatic_from_index(
                pred_v, mode="add_aromatic"
            )
            mol = reconstruct.reconstruct_from_generated(
                pred_pos, pred_atom_type, pred_aromatic
            )
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            logger.warning(f"Reconstruct failed {i}")
            continue
        n_recon_success += 1

        if "." in smiles:  # skip disconnected molecules
            continue
        n_complete += 1

        # --------------------------------------------------------------
        # Convert to SDF / PDBQT for docking
        # --------------------------------------------------------------
        ligand_rdmol = Chem.AddHs(mol, addCoords=True)
        sdf_path = Path(result_path) / f"ligand_{i}.sdf"
        pdbqt_path = sdf_path.with_suffix(".pdbqt")

        Chem.SDWriter(str(sdf_path)).write(ligand_rdmol)
        PrepLig(str(sdf_path), "sdf").get_pdbqt(str(pdbqt_path))

        # --------------------------------------------------------------
        # Chemical and docking evaluation
        # --------------------------------------------------------------
        try:
            chem_results = scoring_func.get_chem(mol)

            vina_res = {
                mode: evaluate_single_ligand(
                    ligand_pdbqt=str(pdbqt_path),
                    protein_pdbqt=protein_pdbqt,
                    center=center,
                    box_size=box_size,
                    mode=mode,
                    exhaustiveness=exhaustiveness,
                )
                for mode in ["score_only", "minimize", "dock"]
            }
            n_eval_success += 1
        except Exception as e:
            logger.warning(f"Chemistry/docking failed {i}: {e}")
            continue

        # Statistics for successful ones
        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist
        success_pair_dist += pair_dist
        success_atom_types += Counter(pred_atom_type)

        results.append(
            {
                "mol": mol,
                "smiles": smiles,
                "ligand_filename": str(sdf_path),
                "pred_pos": pred_pos,
                "pred_v": pred_v,
                "chem_results": chem_results,
                "vina": vina_res,
            }
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    logger.info(f"Evaluate done! {num_samples} samples in total.")

    validity_dict = {
        "mol_stable": all_mol_stable / num_samples,
        "atm_stable": all_atom_stable / max(all_n_atom, 1),
        "recon_success": n_recon_success / num_samples,
        "eval_success": n_eval_success / num_samples,
        "complete": n_complete / num_samples,
    }
    print_dict(validity_dict, logger)

    # Bond-length statistics
    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(
        c_bond_length_profile
    )
    logger.info("JS bond distances of complete mols: ")
    print_dict(c_bond_length_dict, logger)

    success_pair_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_profile)
    print_dict(success_js_metrics, logger)
    if save:
        eval_bond_length.plot_distance_hist(
            success_pair_profile,
            metrics=success_js_metrics,
            save_path=os.path.join(result_path, "pair_dist_hist.png"),
        )

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info("Atom type JS: %.4f" % atom_type_js)

    logger.info(
        "Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d",
        n_recon_success,
        n_complete,
        len(results),
    )

    # Chemical scores summary
    qed = [r["chem_results"]["qed"] for r in results]
    sa = [r["chem_results"]["sa"] for r in results]
    logger.info("QED:   Mean: %.3f Median: %.3f", np.mean(qed), np.median(qed))
    logger.info("SA:    Mean: %.3f Median: %.3f", np.mean(sa), np.median(sa))

    vina_score_only = [r["vina"]["score_only"]["affinity"] for r in results]
    vina_min = [r["vina"]["minimize"]["affinity"] for r in results]
    vina_dock = [r["vina"]["dock"]["affinity"] for r in results]
    logger.info(
        "Vina Score:  Mean: %.3f Median: %.3f",
        np.mean(vina_score_only),
        np.median(vina_score_only),
    )
    logger.info(
        "Vina Min  :  Mean: %.3f Median: %.3f", np.mean(vina_min), np.median(vina_min)
    )
    logger.info(
        "Vina Dock :  Mean: %.3f Median: %.3f", np.mean(vina_dock), np.median(vina_dock)
    )

    # Save Vina affinity histograms
    if save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        data_sets = [vina_score_only, vina_min, vina_dock]
        titles = ["score_only", "minimize", "dock"]
        for ax, data, title in zip(axes, data_sets, titles):
            ax.hist(data, bins=20, color="skyblue", edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Affinity (kcal/mol)")
            ax.set_ylabel("Count")
        fig.tight_layout()
        plt.savefig(os.path.join(result_path, "vina_affinity_hist.png"))
        plt.close(fig)

    # Ring distribution
    print_ring_ratio([r["chem_results"]["ring_size"] for r in results], logger)

    # ------------------------------------------------------------------
    # Save top ligands by docking score (more negative == better)
    # ------------------------------------------------------------------
    if results:  # proceed only if there are evaluated ligands
        # sort by Vina dock affinity (ascending, because lower = better)
        sorted_by_affinity = sorted(
            results, key=lambda x: x["vina"]["dock"]["affinity"]
        )
        top_k = sorted_by_affinity[: min(save_top_k, len(sorted_by_affinity))]

        top_dir = Path(result_path) / f"top{save_top_k}"
        top_dir.mkdir(parents=True, exist_ok=True)

        combined_sdf_path = top_dir / f"top{save_top_k}_ligands.sdf"
        sdf_writer = Chem.SDWriter(str(combined_sdf_path))

        csv_path = top_dir / f"top{save_top_k}_affinity.csv"
        with open(csv_path, "w", newline="") as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(["rank", "smiles", "affinity", "sdf_filename"])

            for rank, res in enumerate(top_k, start=1):
                affinity = res["vina"]["dock"]["affinity"]
                smiles = res["smiles"]
                sdf_source = Path(res["ligand_filename"])
                sdf_dest = top_dir / sdf_source.name

                # copy individual SDF
                try:
                    shutil.copy(sdf_source, sdf_dest)
                except Exception as e:
                    logger.warning(f"Failed to copy {sdf_source} -> {sdf_dest}: {e}")

                # write to combined SDF with informative title
                mol = Chem.AddHs(res["mol"], addCoords=True)
                mol.SetProp("_Name", f"rank_{rank}_affinity_{affinity:.3f}")
                sdf_writer.write(mol)

                csv_writer.writerow([rank, smiles, affinity, sdf_dest.name])

        sdf_writer.close()
        logger.info(
            "Saved top %d ligands (by Vina dock affinity) to %s",
            len(top_k),
            top_dir,
        )

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Inspect and evaluate generated ligands")
    p.add_argument("--ligands", required=True, help="Path to ligands .pt file")
    p.add_argument("--protein", required=True, help="Prepared protein PDBQT file")
    p.add_argument(
        "--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z")
    )
    p.add_argument(
        "--box-size", nargs=3, type=float, default=[10, 10, 10], metavar=("X", "Y", "Z")
    )
    p.add_argument(
        "--result_path",
        default="results/evaluation",
        help="Directory for intermediate ligand files",
    )
    p.add_argument(
        "--save", action="store_true", help="Save intermediate files and plots"
    )
    p.add_argument(
        "--save_top_k", type=int, default=100, help="Number of top ligands to save"
    )
    p.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness")
    return p.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        ligands_path=args.ligands,
        protein_pdbqt=args.protein,
        center=args.center,
        box_size=args.box_size,
        result_path=args.result_path,
        exhaustiveness=args.exhaustiveness,
        save_top_k=args.save_top_k,
        save=args.save,
    )


if __name__ == "__main__":
    raise SystemExit(main())
