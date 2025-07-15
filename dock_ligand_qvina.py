import argparse
import subprocess

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

VINA_EXHAUSTIVENESS = 128


def smiles_to_3d_pdb(smiles, output_pdb):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, output_pdb)


def convert_to_pdbqt(input_file, output_file):
    subprocess.run(["obabel", input_file, "-O", output_file], check=True)


def prepare_receptor(receptor_pdb, output_pdbqt):
    subprocess.run(
        ["prepare_receptor4.py", "-r", receptor_pdb, "-o", output_pdbqt], check=True
    )


def run_qvina(receptor_pdbqt, ligand_pdbqt, center, size, out_pdbqt, log_file):
    command = [
        "qvina2",
        "--receptor",
        receptor_pdbqt,
        "--ligand",
        ligand_pdbqt,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--out",
        out_pdbqt,
        "--log",
        log_file,
        "--exhaustiveness",
        str(VINA_EXHAUSTIVENESS),
    ]
    subprocess.run(command, check=True)


def generate_3d_conformers(smiles, ligand_pdb):
    subprocess.run(["obabel", "-:" + smiles, "-O", ligand_pdb, "--gen3d"])


def convert_pdbqt_to_pdb(input_pdbqt, output_pdb):
    subprocess.run(["obabel", input_pdbqt, "-O", output_pdb], check=True)


def main():
    parser = argparse.ArgumentParser(description="Automated qVina docking from SMILES")
    parser.add_argument(
        "--receptor", required=True, help="Path to receptor .pdbqt file"
    )
    parser.add_argument("--df", required=True, help="Ligand SMILES csv")
    parser.add_argument(
        "--center", nargs=3, type=float, required=True, help="Docking box center x y z"
    )
    parser.add_argument(
        "--size", nargs=3, type=float, required=True, help="Docking box size x y z"
    )
    parser.add_argument("--path", required=True, help="output path")
    # parser.add_argument("--out", default="docked_ligand.pdb", help="Output PDB file for docked ligand")

    args = parser.parse_args()

    ligand_pdb = "ligand.pdb"
    ligand_pdbqt = "ligand.pdbqt"
    receptor_pdbqt = args.receptor

    df = pd.read_csv(args.df)
    # print("[1] Preparing receptor...")
    # prepare_receptor(args.receptor, receptor_pdbqt)

    for smid, smiles in tqdm(enumerate(df.SMILES)):
        # smiles_to_3d_pdb(smiles.split(".")[0], ligand_pdb)
        try:
            generate_3d_conformers(smiles.split(".")[0], ligand_pdb)
            convert_to_pdbqt(ligand_pdb, ligand_pdbqt)
            out = f"{args.path}/docked_ligand_{smid}.pdb"
            log_file = f"{args.path}/docking_{smid}.log"
            docking_output_pdbqt = "docked_ligand.pdbqt"
            run_qvina(
                receptor_pdbqt,
                ligand_pdbqt,
                args.center,
                args.size,
                docking_output_pdbqt,
                log_file,
            )

            convert_pdbqt_to_pdb(docking_output_pdbqt, out)
        except Exception as _:
            pass


if __name__ == "__main__":
    main()
