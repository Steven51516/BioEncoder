from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem import rdmolfiles, rdmolops
from functools import partial
import dgl
import torch


def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
    if mol is None:
        print('Invalid mol found')
        return None


    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g


def smile_to_bigraph(x, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(x)
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)


def construct_bigraph_from_mol(mol, add_self_loop=False):
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g


def smile_to_complete_graph(smile, add_self_loop=False,
                          node_featurizer=None,
                          edge_featurizer=None,
                          canonical_atom_order=True,
                          explicit_hydrogens=False,
                          num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smile)
    return mol_to_graph(mol,
                        partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)







def construct_complete_graph_from_mol(mol, add_self_loop=False):
    """Construct a complete graph with topology only for the molecule

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the molecule
    """
    num_atoms = mol.GetNumAtoms()
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                src.append(i)
                dst.append(j)
    g = dgl.graph((torch.IntTensor(src), torch.IntTensor(dst)), idtype=torch.int32)

    return g


def edge_dist_featurizer(mol, add_self_loop=True):
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate a 3D conformation for the molecule

    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                pos_i = mol.GetConformer().GetAtomPosition(i)  # position of atom i
                pos_j = mol.GetConformer().GetAtomPosition(j)  # position of atom j

                distance = torch.norm(torch.tensor([pos_i.x, pos_i.y, pos_i.z]) - torch.tensor([pos_j.x, pos_j.y, pos_j.z]))
                feats.append(distance)

    return {'d': torch.stack(feats).unsqueeze(-1).float()}