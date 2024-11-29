from ase import Atoms
from ase.io import write, read
from ase.calculators.singlepoint import SinglePointCalculator
atomic_energies={"C": "-1025.34896263954", 
                 "O": "-2034.97553102427",
                 "N": "-1479.90402789208",
                 "H": "-13.5576149854593",}

symbols = ['C', 'O', 'N', 'H']
def substractSelfEnergy(atoms):
    total_self_energy = 0
    for symbol in symbols:
        num_atoms = len([atom for atom in atoms if atom.symbol == symbol])
        self_energy = num_atoms * float(atomic_energies[symbol])
        total_self_energy += self_energy
    return total_self_energy 

atoms_list = read('engrad_all_iter123_100K_MD_7571.extxyz', index=":")
fl = open('test.csv', 'w')
for atoms in atoms_list:
    total_energy = atoms.get_potential_energy()
    self_energy = substractSelfEnergy(atoms)
    cohesive_energy = total_energy-self_energy
  
    print(cohesive_energy/len(atoms), file=fl)
    
    print(atoms.get_potential_energy())
    atoms.calc = SinglePointCalculator(atoms, energy=cohesive_energy, forces=atoms.get_forces())
    atoms.get_potential_energy()

    write('cohesive_engrad_all_iter123_100K_MD_7571.extxyz', atoms, append=True)


    
