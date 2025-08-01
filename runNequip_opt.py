#

import torch
from nequip.ase import NequIPCalculator
from ase.io import read, write
import tqdm
import argparse
from ase.optimize import BFGS



def getBoolStr(string):
    string = string.lower()
    if "true" in string or "yes" in string:
        return True
    elif "false" in string or "no" in string:
        return False
    else:
        print("%s is bad input!!! Must be Yes/No or True/False" %string)
        sys.exit(1)



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="")
parser.add_argument("-model_path", type=str, required=True, help="")
parser.add_argument("-opt", type=str, required=True, help="")
args = parser.parse_args()

device = "cuda"


extxyz_path = args.extxyz_path
model_path = args.model_path
opt = getBoolStr(args.opt)

#  file_base = extxyz_path.split("/")[-1].split(".")[0]
file_base = model_path.split("/")[-1].split(".")[0]

calc = NequIPCalculator.from_deployed_model(
    model_path=model_path,
    device=device,
)

atoms_list = read(extxyz_path, index=":")

fl_enegies = open(f"{file_base}_qm_model_opt{args.opt}_energeis.csv", "w")
fl_enegies.write(f"Name,NNP_Energy\n")

#  while n_sample <= 250:
for i in tqdm.trange(0,len(atoms_list), 1):
#for i in range(0, len(atoms_list), 1):
    atoms = atoms_list[i]
    try:
        label = atoms.info["label"]
    except:
        label = f"frame_{i}"

    atoms.calc = calc
    if opt:
        dyn = BFGS(atoms)
        dyn.run(fmax=0.001)
    model_energy = atoms.get_potential_energy()
    #energy_diff = model_energy - qm_enery


    #  diff_forces = abs(qm_forces - model_forces)

    fl_enegies.write(f"{label},{model_energy}\n")
    fl_enegies.flush()


