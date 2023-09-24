import os
import os.path as osp
import re
from tqdm import tqdm

from sklearn.utils import shuffle
import numpy as np

import torch
from torch_geometric.data import (Dataset, DataLoader, InMemoryDataset, Data, download_url, extract_gz)
import rdkit.Chem.AllChem as AllChem

x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}





def moleculeNet(smiles):
    import rdkit.Chem as Chem
    if smiles == '[C@@H]3(C1=CC=C(Cl)C=C1)[C@H]2CC[C@@H](C2)C34CCC(=N4)N5CCOCC5':
        return None
    if smiles.count('%') >= 15 or smiles.count('@') >= 15:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n_atoms = len(mol.GetAtoms())  # 분자의 원자 개수 1개면 skip
    if n_atoms == 1:
        return None

    # Get Atom Number
    zs = []
    xs = []
    for atom in mol.GetAtoms():
        z = []
        z.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        zs.append(z)

        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(
            str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    z = torch.tensor(zs, dtype=torch.long).view(-1)


    try:
        AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=22, numThreads=0, useRandomCoords=True)
    except:
        return None
    try:
        li = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=22222)  # MMFF의 경우
    except:
        return None
    li = [t[1] for t in li]
    sortidx = torch.argsort(torch.tensor(li))
    if len(sortidx) == 5:
        maxidx1, maxidx2, maxidx3, maxidx4 = int(sortidx[-1]), int(sortidx[-2]), int(sortidx[-3]), int(
            sortidx[-4])
    elif len(sortidx) == 4:
        maxidx1, maxidx2, maxidx3, maxidx4 = int(sortidx[-1]), int(sortidx[-1]), int(sortidx[-2]), int(
            sortidx[-3])
    elif len(sortidx) == 3:
        maxidx1, maxidx2, maxidx3, maxidx4 = int(sortidx[-1]), int(sortidx[-1]), int(sortidx[-2]), int(
            sortidx[-2])
    else:
        return None
    minidx = int(sortidx[0])
    min_energy = li[minidx]
    max1_energy, max2_energy, max3_energy, max4_energy = li[maxidx1], li[maxidx2], li[maxidx3], li[maxidx4]
    minpos = mol.GetConformer(minidx).GetPositions()
    max1pos = mol.GetConformer(maxidx1).GetPositions()
    max2pos = mol.GetConformer(maxidx2).GetPositions()
    max3pos = mol.GetConformer(maxidx3).GetPositions()
    max4pos = mol.GetConformer(maxidx4).GetPositions()

    minpos_mmff = torch.tensor(minpos, dtype=torch.float)
    max1pos_mmff = torch.tensor(max1pos, dtype=torch.float)
    max2pos_mmff = torch.tensor(max2pos, dtype=torch.float)
    max3pos_mmff = torch.tensor(max3pos, dtype=torch.float)
    max4pos_mmff = torch.tensor(max4pos, dtype=torch.float)
    # atom 좌표를 제대로 생성하지 못할 때
    if 0.0 in minpos_mmff:
        return None
    data = Data(pos=minpos_mmff, max1pos_mmff=max1pos_mmff, max2pos_mmff=max2pos_mmff,
                max3pos_mmff=max3pos_mmff, max4pos_mmff=max4pos_mmff,
                min_energy=min_energy, max1_energy=max1_energy, max2_energy=max2_energy,
                max3_energy=max3_energy, max4_energy=max4_energy,
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                z=z,  smiles=smiles)
    return data








