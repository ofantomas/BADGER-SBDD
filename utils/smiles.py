from typing import Dict, Sequence, List, Union

import torch
from rdkit import Chem

from utils.transforms import get_atomic_number_from_index, is_aromatic_from_index
from utils.reconstruct import reconstruct_from_generated


# Using explicit Union for Python <3.10 compatibility
def compose_smiles(
    ligand: Dict[str, Union[torch.Tensor, Sequence]],
    *,
    atom_enc_mode: str = "add_aromatic",
) -> str:
    """Convert **one** generated ligand (``{'pos': pos, 'v': v}``) into a SMILES string.

    Parameters
    ----------
    ligand : dict
        Dictionary containing
        * ``pos`` – ``(N, 3)`` Cartesian coordinates (torch.Tensor or array-like)
        * ``v``   – ``(N,)`` atom-type indices produced by the diffusion model.
    atom_enc_mode : str, optional
        Encoding scheme that was used for the atom types during training.
        Should be identical to the value passed when building ``FeaturizeLigandAtom``
        (defaults to ``"add_aromatic"`` which is the setting of the released checkpoints).

    Returns
    -------
    str
        Canonical SMILES string generated with RDKit.
    """
    if "pos" not in ligand or "v" not in ligand:
        raise KeyError("Ligand dictionary must contain 'pos' and 'v' keys.")

    # --- Prepare tensors ---------------------------------------------------
    pos = ligand["pos"]
    v = ligand["v"]

    # Convert to torch.Tensor if necessary (keeps existing tensor object otherwise)
    pos = torch.as_tensor(pos, dtype=torch.float32, device="cpu")
    v = torch.as_tensor(v, dtype=torch.long, device="cpu")

    # Map atom-type indices -> atomic numbers (& aromaticity flags, if used)
    atomic_nums: List[int] = get_atomic_number_from_index(v, mode=atom_enc_mode)
    if atom_enc_mode == "add_aromatic":
        aromatic_flags: List[bool] = is_aromatic_from_index(v, mode=atom_enc_mode)
        basic_mode = False
    else:
        aromatic_flags = None
        basic_mode = True

    # Reconstruct 3D molecule then convert to SMILES
    rd_mol = reconstruct_from_generated(
        xyz=pos.cpu().tolist(),
        atomic_nums=atomic_nums,
        aromatic=aromatic_flags,
        basic_mode=basic_mode,
    )
    # Generate canonical SMILES (without stereochemistry isomer info)
    smiles = Chem.MolToSmiles(rd_mol)
    return smiles
