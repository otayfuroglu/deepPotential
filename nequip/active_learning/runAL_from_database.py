#

from nequip.ase import NequIPCalculator
from ase.io import read, write
import tqdm, os
import argparse


def load_model(model_path):
    return NequIPCalculator.from_deployed_model(
       model_path=model_path,
       device=device,
    )   



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-struc_dir", type=str, required=True, help="")
parser.add_argument("-model1_path", type=str, required=True, help="")
parser.add_argument("-model2_path", type=str, required=True, help="")
args = parser.parse_args()

#  T = 300
#  it = "iter4"

device = "cuda"

struc_dir = args.struc_dir
model1_path = args.model1_path
model2_path = args.model2_path
calc1 = load_model(model1_path)
calc2 = load_model(model2_path)


#atoms_list = read(extxyz_path, index=":")
#atoms_list = [read(f"{struc_dir/flname}")for flname in os.listdir(fldir)] 


fl = open(f"{struc_dir}_model1_model2_energes.csv", "w")
fl.write(f"index,e_NNP1,e_NNP2,e_diff\n")

#  while n_sample <= 250:
for flname in tqdm.tqdm(os.listdir(struc_dir)):
    #  for i in range(0, len(atoms_list), 1):
    atoms = read(f"{struc_dir}/{flname}")
    atoms.calc = calc1
    e1 = atoms.get_potential_energy() / len(atoms)
    atoms.calc = calc2
    e2 = atoms.get_potential_energy() / len(atoms)
    e_diff = abs(e2 - e1)

    fl.write(f"{flname},{e1},{e2},{e_diff}\n")
    fl.flush()

    if e_diff >= 0.002:
        atoms.info["label"] = flname
        write(f"{struc_dir}_AL_from_database.extxyz", atoms, append=True)
fl.close()
