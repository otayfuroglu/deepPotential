from rdkit import Chem
from rdkit.Chem import SDWriter
import argparse
import os

os.makedirs("sdfFiles", exist_ok=True)

parser = argparse.ArgumentParser(description="")
parser.add_argument("-sdfpath", type=str, required=False, help="give hdf5 file base")
args = parser.parse_args()
sdfpath = args.sdfpath

supplier = Chem.SDMolSupplier(sdfpath)

for i, mol in enumerate(supplier):
    if mol is None:
        print(f"Warning: Molecule {i + 1} could not be read and was skipped.")
        continue

    mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"molecule_{i + 1}"
    output_sdf = os.path.join("sdfFiles", f"{mol_id}.sdf")
    writer = SDWriter(output_sdf)
    writer.write(mol)
    writer.close()

