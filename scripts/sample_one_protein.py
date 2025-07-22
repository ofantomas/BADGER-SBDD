#!/usr/bin/env python
"""sample_one_protein.py
Generate ligand samples for a single protein (or pocket) PDB file using the
DecompDiff diffusion model.  The script follows the SAME hyper-parameter and
transform pipeline as `sample_diffusion_decomp.py`, but instead of iterating
through a dataset it takes one PDB (+ optional prior files) directly from the
user.

Example
-------
python sample_one_protein.py configs/sampling_drift.yml \
       --pdb pockets/6ed6_clean_pocket_15.pdb \
       --center 26.9098 47.024 52.50861 \
       --outdir results/ --num 256 --device cuda:0
"""

import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.transforms import Compose

sys.path.append(os.getcwd())

import utils.misc as misc
import utils.prior as utils_prior
import utils.transforms as trans
from models.classifier import Classifier
from models.decompdiff import DecompScorePosNet3D
from scripts.sample_diffusion_decomp import sample_diffusion_ligand_decomp
from utils import data as utils_data
from utils.data import PDBProtein, ProteinLigandData, torchify_dict

###########################################################################
# Helpers for constructing a dummy ligand                                #
###########################################################################


def build_dummy_ligand(center: List[float]):
    """Return a *single-atom* ligand dictionary placed at *center* (Å)."""
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
    """Parse the pocket PDB and attach a dummy ligand so the model has
    something to start with.  All additional attributes required by DecompDiff
    transforms are added here.
    """
    protein = PDBProtein(pdb_path)
    protein_dict = torchify_dict(protein.to_dict_atom())
    full_protein_pos = protein_dict["pos"].float()

    ligand_dict = torchify_dict(build_dummy_ligand(center))

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict, ligand_dict=ligand_dict
    )

    # PyG Batch cannot handle list/str attributes when it tries to compute
    # incremental shifts. They are not used during diffusion, so we drop them.
    for attr in [
        "protein_atom_name",
        "protein_molecule_name",
        "ligand_hybridization",
        "ligand_nbh_list",
    ]:
        if hasattr(data, attr):
            delattr(data, attr)

    # Decomposition meta-data (single arm + scaffold)
    data.ligand_atom_mask = torch.tensor([0], dtype=torch.long)

    center_t = torch.tensor(center, dtype=torch.float32)
    dists = torch.norm(data.protein_pos - center_t, dim=1)
    pocket_mask = dists < 5.0
    if pocket_mask.sum() == 0:
        pocket_mask = torch.zeros_like(dists, dtype=torch.bool)

    data.pocket_atom_masks = pocket_mask.unsqueeze(0)

    data.num_arms = 1
    data.num_scaffold = 1  # keep 1 to satisfy downstream cat operations

    return data, full_protein_pos


###########################################################################
# Main                                                                    #
###########################################################################

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # CLI (inherits almost all arguments from sample_diffusion_decomp)
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML config file")
    parser.add_argument("--pdb", required=True, help="Pocket PDB file")
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Binding-site center (Å)",
    )
    parser.add_argument("--outdir", type=str, default="./outputs_one_protein")
    parser.add_argument("--outfile", type=str, default="ligands.pt")

    # Re-expose all hyper-parameters that sample_diffusion_decomp understands
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--prior_mode",
        type=str,
        choices=["subpocket", "ref_prior", "beta_prior"],
        default=None,
    )
    parser.add_argument(
        "--num_atoms_mode",
        type=str,
        choices=["prior", "ref", "ref_large"],
        default="prior",
    )
    parser.add_argument(
        "--natom_models_path", type=str, default="./pregen_info/natom_models.pkl"
    )
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_name", type=str, default="one_protein_sampling")
    # Allow overriding sample parameters
    parser.add_argument("--num", type=int, default=None, help="Number of samples")

    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(8)

    # ---------------------------------------------------------------------
    # Load config + checkpoint
    # ---------------------------------------------------------------------
    config = misc.load_config(args.config)
    misc.seed_all(config.sample.seed)

    # ckpt_path = args.ckpt_path if args.ckpt_path else config.model.checkpoint
    ckpt_path = "/mnt/5tb/tsypin/EyeDrops/BADGER-SBDD/checkpoints/pretrained_models/uni_o2_bond.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if "train_config" in config.model:
        ckpt["config"] = misc.load_config(config.model["train_config"])

    print(ckpt["config"])

    # ---------------------------------------------------------------------
    # Transforms (identical to sample_diffusion_decomp)
    # ---------------------------------------------------------------------
    cfg_transform = ckpt["config"].data.transform
    protein_featurizer = trans.FeaturizeProteinAtom()

    ligand_atom_mode = cfg_transform.ligand_atom_mode
    ligand_bond_mode = cfg_transform.ligand_bond_mode
    max_num_arms = cfg_transform.max_num_arms

    print(ligand_bond_mode)

    ligand_featurizer = trans.FeaturizeLigandAtom(
        ligand_atom_mode, prior_types=ckpt["config"].model.get("prior_types", False)
    )
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=getattr(ckpt["config"].data.transform, "add_ord_feat", True),
    )
    transform_prot = Compose([protein_featurizer])

    # For new pocket sampling
    prior_mode = "subpocket"
    init_transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=prior_mode),
        decomp_indicator,
    ]
    if getattr(ckpt["config"].model, "bond_diffusion", False):
        init_transform_list.append(
            trans.FeaturizeLigandBond(mode=ligand_bond_mode, set_bond_type=False)
        )
    init_transform = Compose(init_transform_list)

    # ---------------------------------------------------------------------
    # Build *single* data object
    # ---------------------------------------------------------------------
    data, full_protein_pos = build_data(args.pdb, args.center)

    # Setup prior information if required
    if prior_mode == "ref_prior":
        utils_prior.compute_golden_prior_from_data(data)
    elif prior_mode == "beta_prior":
        pocket_id = os.path.splitext(os.path.basename(args.pdb))[0]
        beta_pkl = os.path.join(args.beta_prior_path, f"{pocket_id}.pkl")
        utils_prior.substitute_golden_prior_with_beta_prior(
            data, beta_prior_path=beta_pkl
        )
    # subpocket needs no extra processing
    data = transform_prot(data)  # Protein features last (uses new attributes)
    # ---------------------------------------------------------------------
    # Load classifier
    # ---------------------------------------------------------------------
    clf_ckpt = torch.load(
        "checkpoints/load_ckpt/decompdiff_single_constraints/ckpt.pt",
        map_location=args.device,
        weights_only=False,
    )

    classifier = Classifier(
        clf_ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.protein_feature_dim
        + decomp_indicator.protein_feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.ligand_feature_dim
        + decomp_indicator.ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        device=args.device,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob,
    ).to(args.device)
    classifier.load_state_dict(clf_ckpt["model"], strict=False)
    classifier.eval()

    # ---------------------------------------------------------------------
    # Load diffusion model
    # ---------------------------------------------------------------------
    model = DecompScorePosNet3D(
        ckpt["config"].model,
        protein_atom_feature_dim=protein_featurizer.protein_feature_dim
        + decomp_indicator.protein_feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.ligand_feature_dim
        + decomp_indicator.ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob,
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ---------------------------------------------------------------------
    # Aux configs (num-atoms predictors etc.)
    # ---------------------------------------------------------------------
    with open(config.sample.arms_num_atoms_config, "rb") as f:
        arms_num_atoms_config = pickle.load(f)
    with open(config.sample.scaffold_num_atoms_config, "rb") as f:
        scaffold_num_atoms_config = pickle.load(f)

    if args.num is not None:
        num_samples = args.num
    else:
        num_samples = config.sample.num_samples

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    results = sample_diffusion_ligand_decomp(
        model,
        data,
        full_protein_pos,
        init_transform=init_transform,
        num_samples=num_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        num_atoms_mode=args.num_atoms_mode,
        arms_natoms_config=arms_num_atoms_config,
        scaffold_natoms_config=scaffold_num_atoms_config,
        atom_enc_mode=ligand_atom_mode,
        bond_fc_mode=ligand_bond_mode,
        energy_drift_opt=config.sample.energy_drift,
        prior_mode=prior_mode,
        atom_prior_probs=ligand_featurizer.atom_types_prob,
        bond_prior_probs=ligand_featurizer.bond_types_prob,
        natoms_config=args.natom_models_path,
        context=config.sample.delta_context,
        scale_factor=config.sample.scale_factor,
        classifier=classifier,
        clip=float(config.sample.clip),
        enable_wandb=args.wandb,
    )

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.outfile)
    torch.save(
        {
            "results": results,
        },
        out_path,
    )
    print(f"Saved generated ligands to {out_path}")
