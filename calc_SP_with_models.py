#
from nequip.ase import NequIPCalculator
import pandas as pd
import torch
#  from ase.db import connect
#  from scipy import stats
#  from matplotlib import pyplot as plt

from multiprocessing import Pool, current_process, set_start_method

import tqdm
from ase.io import write, read
from ase.db import connect

import numpy as np

import os, sys, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

import subprocess
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-mof_num", "--mof_num",
#                      type=int, required=True,
                    #  help="..")
parser.add_argument("-val_type", "--val_type",
                    type=str, required=True,
                    help="..")
parser.add_argument("-data_path", "--data_path",
                    type=str, required=True,
                    help="..")
parser.add_argument("-model_path", "--model_path",
                    type=str, required=True,
                    help="..")
parser.add_argument("-RESULT_DIR", "--RESULT_DIR",
                    type=str, required=True,
                    help="..")
parser.add_argument("-nprocs", "--nprocs",
                    type=int, required=True,
                    help="..")
args = parser.parse_args()

#  mof_num = args.mof_num
mode = args.val_type
RESULT_DIR = args.RESULT_DIR
device = "cuda"


def get_fRMS(forces):
    """
    Args:
    forces(3D torch tensor)
    retrun:
    force RMS (zero D torch tensor)

     Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print("Cartsian Forces: RMS" , torch.sqrt(((A -A.mean(axis=0))**2).mean()) #RMS force
    Output:  Cartsian Forces: RMS     5.009549113310641e-06
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return torch.sqrt(((forces - forces.mean(axis=0))**2).mean())


def get_fmax_atomic(forces):
    """
    Args:
    forces(3D torch tensor)
    retrun:
    maximum atomic force (zero D torch tensor)

    Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print(("Cartsian Forces: Max atomic force", (A**2).sum(1).max()**5 # Maximum atomic force (fom ASE)
    Output: Cartsian Forces: Max atomic forces 0.0000
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return (forces**2).sum(1).max()**5 # Maximum atomic force (fom ASE)


def get_fmax_component(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent(zero D torch tensor)

    Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print("Cartsian Forces: Max", get_fmax_component(A)
    Output:Cartsian Forces: Max -1.831199915613979e-05
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return forces[abs_idxs[0], abs_idxs[1]].item()


def get_fmax_idx(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent indices (zero 1D torch tensor)
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return abs_idxs


def get_fmax_componentFrom_idx(forces, fmax_component_idx):
    """
    xxx
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return forces[fmax_component_idx[0], fmax_component_idx[1]].item()


def getSPEneryForces(idx):
    #  os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 2)

    #  creating working directory for each processors doe to avoid confliction
    #  current = current_process()
    #  proc_dir = RESULT_DIR + "/tmp_%s" % current._identity
    #  if not os.path.exists(proc_dir):
    #      os.mkdir(proc_dir)
    #  os.chdir(proc_dir)

    #  calculator = NequIPCalculator.from_deployed_model(model_path=args.model_path, device=device)

    #  data = connect(args.data_path)
    #  row = data.get(idx+1)  # +1 because of index starts 1
    file_names = "frame_%d" %idx
    #  write("test_atom_%d.xyz" %idx, mol)
    mol = data[idx]
    n_atoms = len(mol)

    qm_energy = mol.get_potential_energy()
    qm_forces = mol.get_forces()
    qm_fmax = (qm_forces**2).sum(1).max()**0.5

    # first get fmax indices than get qm_fmax and n2p2 fmax component
    qm_fmax_component_idx = get_fmax_idx(qm_forces)
    qm_fmax_component = get_fmax_componentFrom_idx(qm_forces,
                                                   qm_fmax_component_idx)

    mol.pbc = True
    mol.set_calculator(calculator)

    n2p2_energy = mol.get_potential_energy()
    n2p2_forces = mol.get_forces()
    n2p2_fmax = (n2p2_forces**2).sum(1).max()**0.5  # Maximum atomic force (fom ASE).
    n2p2_fmax_component = get_fmax_componentFrom_idx(n2p2_forces,
                                                           qm_fmax_component_idx)

    energy_err = qm_energy - n2p2_energy
    fmax_err = qm_fmax - n2p2_fmax
    fmax_component_err = qm_fmax_component - n2p2_fmax_component
    # calculate error for all forces on atoms
    fall_err = torch.flatten(torch.from_numpy(qm_forces) - n2p2_forces)
    #  fall_err = qm_forces.flatten - n2p2_forces.flatten


    energy_values = [file_names,
                     qm_energy,
                     n2p2_energy,
                     energy_err,
                     qm_energy/n_atoms,
                     n2p2_energy/n_atoms,
                     energy_err/n_atoms,
                    ]

    fmax_values = [file_names,
                     qm_fmax,
                     n2p2_fmax,
                     fmax_err,
                    ]

    fmax_component_values = [file_names,
                     qm_fmax_component,
                     n2p2_fmax_component,
                     fmax_component_err,
                    ]

    df_data_energy.loc[0] = energy_values # rewrite on first row at each steps
    df_data_fmax.loc[0] = fmax_values
    df_data_fmax_component.loc[0] = fmax_component_values

    df_data_fall = pd.DataFrame(columns=["FileNames", "qm_SP_F_all", "n2p2_SP_F_all", "F_all_Error"]) # reset df
    df_data_fall["FileNames"] = [file_names] * len(qm_forces.flatten())
    df_data_fall["qm_SP_F_all"] = qm_forces.flatten()
    df_data_fall["n2p2_SP_F_all"] = n2p2_forces.flatten()
    df_data_fall["F_all_Error"] = fall_err.numpy()

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), mode="a", header=False, float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), mode="a", header=False, float_format='%.6f')
    df_data_fmax_component.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fmax_component), mode="a", header=False, float_format='%.6f')
    df_data_fall.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fall), mode="a", header=False, float_format='%.6f')

def run():
    #  n_data = len(data)
    #  idxs = range(n_data)

    for idx in tqdm.tqdm(idxs):
        getSPEneryForces(idx)


def run_multiproc(n_procs):
    n_data = len(data)
    idxs = range(n_data)
    #  idxs = range(200)
    print("Nuber of %s data points: %d" %(mode, len(idxs)))
    #  result_list_tqdm = []
    run_multiprocessing(func=getSPEneryForces,
                                           argument_list=idxs,
                                           num_processes=n_procs)


def run_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []

    # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    for result in tqdm.tqdm(pool.imap_unordered(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    #  return result_list_tqdm


if __name__ == "__main__":

    #  set_start_method('spawn')
    #  data = connect(args.data_path)
    data = read(args.data_path, index=":")
    n_data = len(data)
    idxs = range(n_data)

    calculator = NequIPCalculator.from_deployed_model(model_path=args.model_path, device=device)

    column_names_energy = [
        "FileNames",
        "qm_SP_energies",
        "n2p2_SP_energies",
        "Error",
        "qm_SP_energiesPerAtom",
        "n2p2_SP_energiesPerAtom",
        "ErrorPerAtom",
    ]

    column_names_fmax = [
        "FileNames",
        "qm_SP_fmax",
        "n2p2_SP_fmax",
        "Error",
    ]

    column_names_fmax_component = [
        "FileNames",
        "qm_SP_fmax_component",
        "n2p2_SP_fmax_component",
        "Error",
    ]

    csv_file_name_energy = "qm_sch_SP_E_%s.csv" %(mode)
    csv_file_name_fmax = "qm_sch_SP_F_%s.csv" %(mode)
    csv_file_name_fmax_component = "qm_sch_SP_FC_%s.csv" %(mode)
    csv_file_name_fall = "qm_sch_SP_FAll_%s.csv" %(mode)

    df_data_energy = pd.DataFrame(columns=column_names_energy)
    df_data_fmax = pd.DataFrame(columns=column_names_fmax)
    df_data_fmax_component = pd.DataFrame(columns=column_names_fmax_component)
    df_data_fall = pd.DataFrame(columns=["FileNames", "qm_SP_F_all", "n2p2_SP_F_all", "F_all_Error"]) # reset df

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), float_format='%.6f')
    df_data_fmax_component.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fmax_component), float_format='%.6f')
    df_data_fall.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fall), float_format='%.6f')

    #  run_multiprocessing(func=getSPEneryForces, argument_list=idxs, num_processes=arg.nprocs)
    #  run_multiproc(args.nprocs)
    run()

    # remove tmp_ precessor working directory
    #  os.system(f"rm -r {RESULT_DIR}/tmp_*")
