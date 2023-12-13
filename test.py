from BioEncoder.dataset.loader import *
from BioEncoder.encoder.BE import *
from BioEncoder.task.BETrainer import *
import gc
gc.collect()

pdb_ids, drugs, proteins, labels = load_DTI("data/DAVIS/davis.txt", id=True)
BE = BioEncoder()
de = BE.init_drug_encoder("GCN")
pe = BE.init_prot_encoder("GCN", pdb=True)
inter = BE.set_interaction([de, pe], "cat", head=1)
BE.build_model()
trainer = BETrainer(BE)
train, val, test = trainer.prepare_datasets(inputs = [drugs, proteins, pdb_ids], y=labels, split_frac = [0.7, 0.1, 0.2],
                                            input_types=[BE.DRUG, BE.PROT, BE.PROT_PDB])
trainer.train(train, val)












#
#
# import rdkit
# from rdkit import Chem
# from rdkit.Chem import BRICS
#
# # Assign unique identifiers to each atom in the molecule
# def assign_atom_ids(mol):
#     for i, atom in enumerate(mol.GetAtoms()):
#         atom.SetProp('molAtomMapNumber', str(i))
#
# # Function to fragment a molecule using the BRICS method
# def fragment_molecule_brics(mol):
#     brics_mol = BRICS.BreakBRICSBonds(mol)
#     fragment_tuples = Chem.GetMolFrags(brics_mol, asMols=True, sanitizeFrags=False)
#     return fragment_tuples
#
# # Example molecule
# smiles = drugs[0]
# mol = Chem.MolFromSmiles(smiles)
# assign_atom_ids(mol)
#
# # Fragment the molecule
# fragments = fragment_molecule_brics(mol)
#
# # Create a list to map each atom to its fragment
# atom_to_fragment = [-1] * mol.GetNumAtoms()
#
# # Map each atom in the fragment to the original molecule's atom
# for fragment_idx, frag in enumerate(fragments):
#     for atom in frag.GetAtoms():
#         if atom.HasProp('molAtomMapNumber'):
#             original_atom_idx = int(atom.GetProp('molAtomMapNumber'))
#             atom_to_fragment[original_atom_idx] = fragment_idx
#
# # Print the mapping
# print("Atom index to Fragment index mapping:")
# print(atom_to_fragment)


