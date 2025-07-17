#!/usr/bin/env python
"""sample_one_protein.py
Generate *N* ligand samples for a single protein (or pocket) PDB file using the
pre-trained BADGER-SBDD diffusion model with binding-affinity guidance.

Example
-------
python sample_one_protein.py --pdb 1pxx_pocket.pdb \
    --center 27.116 24.090 14.936 \
    --outdir results_1pxx --num 100 --device cuda:0
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from torch_geometric.transforms import Compose

sys.path.append(os.getcwd())

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.classifier_cur import Classifier
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils import data as utils_data
from utils.data import PDBProtein


def build_dummy_ligand(center: List[float]):
    """Create a single-atom dummy ligand at *center* (required by the model).

    NOTE: This dummy ligand does NOT determine the size of generated molecules!
    The actual molecule sizes are sampled based on pocket statistics during diffusion.
    This is just a placeholder to create a valid input data structure.
    """
    element = np.array([6], dtype=np.int64)  # carbon
    pos = np.array([center], dtype=np.float32)
    bond_index = np.empty((2, 0), dtype=np.int64)
    bond_type = np.empty((0,), dtype=np.int64)
    atom_feature = np.zeros((1, len(utils_data.ATOM_FAMILIES)), dtype=np.int64)
    hybridization = ["SP3"]
    return {
        "element": element,
        "pos": pos,
        "bond_index": bond_index,
        "bond_type": bond_type,
        "atom_feature": atom_feature,
        "hybridization": hybridization,
    }


def build_data(pdb_path: str, center: List[float]):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    ligand_dict = build_dummy_ligand(center)
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pdb_path
    data.ligand_filename = "dummy"
    return data


def main(args, config):
    device = torch.device(args.device)

    # ---------------- Load pretrained diffusion model ----------------
    ckpt = torch.load(
        "checkpoints/pretrained_models/pretrained_diffusion.pt",
        map_location=device,
        weights_only=False,
    )

    # Use the same ligand atom mode as was used during training
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode

    # Transformations identical to training pipeline
    feat_prot = trans.FeaturizeProteinAtom()
    feat_lig = trans.FeaturizeLigandAtom(mode=ligand_atom_mode)
    transform = Compose([feat_prot, feat_lig, trans.FeaturizeLigandBond()])

    data = build_data(args.pdb, args.center)
    data = transform(data)

    model = ScorePosNet3D(
        ckpt["config"].model,
        protein_atom_feature_dim=feat_prot.feature_dim,
        ligand_atom_feature_dim=feat_lig.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------------- Load pretrained classifier ---------------------
    clf_ckpt = torch.load(
        "checkpoints/load_ckpt/targetdiff_single_constraint_egnn/ckpt.pt",
        map_location=device,
        weights_only=False,
    )
    # ckpt["config"].model.num_ffn_head = 256
    clf = Classifier(
        clf_ckpt["config"].model,
        protein_atom_feature_dim=feat_prot.feature_dim,
        ligand_atom_feature_dim=feat_lig.feature_dim,
        device=device,
    ).to(device)
    clf.load_state_dict(clf_ckpt["model"])
    clf.eval()

    # ---------------- Run sampling -----------------------------------
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    pred_pos, pred_v, *_ = sample_diffusion_ligand(
        model,
        data,
        config.sample.num_samples,
        batch_size=args.batch,
        device=device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        context=config.sample.context,
        s=config.sample.s,
        s_v=config.sample.s_v,
        classifier=clf,
        clip=config.sample.clip,
    )

    torch.save({"pos": pred_pos, "v": pred_v}, args.outfile)
    print(f"Saved generated ligands to {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="Protein or pocket PDB file")
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Binding site center coordinates (Å)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/sampling.yml", help="Config file"
    )
    parser.add_argument(
        "--outfile",
        default="results/ligands/ligands.pt",
        help="File name to save outputs",
    )
    parser.add_argument(
        "--batch", type=int, default=10, help="Mini-batch size during sampling"
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device or 'cpu'")
    args = parser.parse_args()

    config = misc.load_config(args.config)
    misc.seed_all(config.sample.seed)

    # Convert center list of str to list[float] here for type consistency
    args.center = [float(x) for x in args.center]
    main(args, config)
