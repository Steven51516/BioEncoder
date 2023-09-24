from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
import dgl
import torch



from graphein.protein.edges.distance import add_hydrogen_bond_interactions
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, meiler_embedding, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor

import torch

import dgl
import torch


import os

from graphein.protein.graphs import process_dataframe, deprotonate_structure, convert_structure_to_centroids, subset_structure_to_atom_type, filter_hetatms, remove_insertions





import os
import traceback
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from biopandas.mmtf import PandasMmtf
from biopandas.pdb import PandasPdb


def add_sequence_distance_edges(
    G: nx.Graph, d: int, name: str = "sequence_edge"
) -> nx.Graph:
    """
    Adds edges based on sequence distance to residues in each chain.

    Eg. if ``d=6`` then we join: nodes ``(1,7), (2,8), (3,9)..``
    based on their sequence number.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :param d: Sequence separation to add edges on.
    :param name: Name of the edge type. Defaults to ``"sequence_edge"``.
    :type name: str
    :return G: Networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:
        # Find chain residues
        chain_residues = [
            (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Subset to only N and C atoms in the case of full-atom
        # peptide bond addition

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            try:
                # Checks not at chain terminus - is this versatile enough?
                if i == len(chain_residues) - d:
                    continue
                # Asserts residues are on the same chain
                cond_1 = (
                    residue[1]["chain_id"]
                    == chain_residues[i + d][1]["chain_id"]
                )
                # Asserts residue numbers are adjacent
                cond_2 = (
                    abs(
                        residue[1]["residue_number"]
                        - chain_residues[i + d][1]["residue_number"]
                    )
                    == d
                )

                # If this checks out, we add a peptide bond
                if (cond_1) and (cond_2):
                    # Adds "peptide bond" between current residue and the next
                    if G.has_edge(i, i + d):
                        G.edges[i, i + d]["kind"].add(name)
                    else:
                        G.add_edge(
                            residue[0],
                            chain_residues[i + d][0],
                            kind={name},
                        )
            except IndexError:
                continue
    return G


def add_peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone as edges to residues in each chain.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :return G: Networkx protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    return add_sequence_distance_edges(G, d=1, name="peptide_bond")
def read_pdb_to_dataframe(
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    model_index: int = 1,
) -> pd.DataFrame:
    """
    Reads PDB file to ``PandasPDB`` object.

    Returns ``atomic_df``, which is a DataFrame enumerating all atoms and
    their cartesian coordinates in 3D space. Also contains associated metadata
    from the PDB file.

    :param path: path to PDB or MMTF file. Defaults to ``None``.
    :type path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to ``None``.
    :type pdb_code: str, optional
    :param uniprot_id: UniProt ID to build graph from AlphaFoldDB. Defaults to
        ``None``.
    :type uniprot_id: str, optional
    :param model_index: Index of model to read. Only relevant for structures
        containing ensembles. Defaults to ``1``.
    :type model_index: int, optional
    :returns: ``pd.DataFrame`` containing protein structure
    :rtype: pd.DataFrame
    """
    if pdb_code is None and path is None and uniprot_id is None:
        raise NameError(
            "One of pdb_code, path or uniprot_id must be specified!"
        )

    if path is not None:
        if isinstance(path, Path):
            path = os.fsdecode(path)
        if (
            path.endswith(".pdb")
            or path.endswith(".pdb.gz")
            or path.endswith(".ent")
        ):
            atomic_df = PandasPdb().read_pdb(path)
        elif path.endswith(".mmtf") or path.endswith(".mmtf.gz"):
            atomic_df = PandasMmtf().read_mmtf(path)
        else:
            raise ValueError(
                f"File {path} must be either .pdb(.gz), .mmtf(.gz) or .ent, not {path.split('.')[-1]}"
            )
    elif uniprot_id is not None:
        atomic_df = PandasPdb().fetch_pdb(
            uniprot_id=uniprot_id, source="alphafold2-v3"
        )
    else:
        atomic_df = PandasPdb().fetch_pdb(pdb_code)

    # atomic_df = atomic_df.get_model(model_index)
    # if len(atomic_df.df["ATOM"]) == 0:
    #     raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])


# def rename_pdb_string(filename):
#     if filename.startswith('ABL1'):
#         # Split the filename into name and extension
#         name, ext = os.path.splitext(filename)
#
#         # Add '-phosphorylated' before the extension
#         new_name = name + "-phosphorylated" + ext
#
#         return new_name
#     else:
#         return filename
#

def create_prot_dgl_graph(pdb_name):
    processing_funcs = [deprotonate_structure, convert_structure_to_centroids, remove_insertions]

    # Read dataframe from PDB
    raw_df = read_pdb_to_dataframe(path="data/DAVIS/pdb/"+pdb_name+".pdb")

    # Apply processing functions
    df = process_dataframe(raw_df, atom_df_processing_funcs=processing_funcs)

    from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph
    from graphein.protein.graphs import annotate_node_metadata, annotate_graph_metadata, annotate_edge_metadata
    g = initialise_graph_with_metadata(protein_df=df,  # from above cell
                                       raw_pdb_df=raw_df,  # Store this for traceability
                                       pdb_code=pdb_name,  # and again
                                       granularity="centroid"  # Store this so we know what kind of graph we have
                                       )
    g = add_nodes_to_graph(g)
    from graphein.protein.graphs import compute_edges
    g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds, add_hydrogen_bond_interactions])
    g = annotate_node_metadata(g, [meiler_embedding, amino_acid_one_hot, expasy_protein_scale, hydrogen_bond_acceptor, hydrogen_bond_donor])
    g_dgl = dgl.from_networkx(g, node_attrs=['amino_acid_one_hot', 'expasy'])  # Convert NetworkX graph to DGL graph
    node_features = []

    for _, data in g.nodes(data=True):
        feature = data.get("meiler").tolist()
        feature1 = data.get("amino_acid_one_hot").tolist()
        feature2 = data.get("expasy").tolist()
        feature3 = data.get("hbond_acceptors").tolist()
        feature4 = data.get("hbond_donors").tolist()
        if feature is not None:
            node_features.append(feature1)
        else:
            node_features.append([
                                     0] * 6)  # Fallback to a default feature if none exists (e.g., a zero vector of the expected length)

    # Convert list of features to a tensor and add to DGL graph
    node_feature_tensor = torch.tensor(node_features)
    g_dgl.ndata['h'] = g_dgl.ndata['amino_acid_one_hot']

    g_dgl = dgl.add_self_loop(g_dgl)
    return g_dgl




def create_prot_esm_dgl_graph(pdb_name):
    from graphein.protein.features.sequence.embeddings import esm_residue_embedding
    processing_funcs = [deprotonate_structure, convert_structure_to_centroids, remove_insertions]

    # Read dataframe from PDB
    raw_df = read_pdb_to_dataframe(path="data/DAVIS/pdb/"+pdb_name+".pdb")

    # Apply processing functions
    df = process_dataframe(raw_df, atom_df_processing_funcs=processing_funcs)

    from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph
    from graphein.protein.graphs import annotate_node_metadata, annotate_graph_metadata, annotate_edge_metadata
    g = initialise_graph_with_metadata(protein_df=df,  # from above cell
                                       raw_pdb_df=raw_df,  # Store this for traceability
                                       pdb_code=pdb_name,  # and again
                                       granularity="centroid"  # Store this so we know what kind of graph we have
                                       )
    g = add_nodes_to_graph(g)
    from graphein.protein.graphs import compute_edges
    g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds, add_hydrogen_bond_interactions])
    g = annotate_node_metadata(g, [amino_acid_one_hot])
    g = annotate_graph_metadata(g, [esm_residue_embedding])
    g_dgl = dgl.from_networkx(g, node_attrs=['esm_embedding_A'])  # Convert NetworkX graph to DGL graph
    node_features = []

    # Convert list of features to a tensor and add to DGL graph
    node_feature_tensor = torch.tensor(node_features)
    g_dgl.ndata['h'] = g_dgl.ndata['esm_embedding_A']
    # g_dgl.edata['e'] = g_dgl.edata['distance']

    edge_features = []
    for u, v, data in g.edges(data=True):
        feature1 = data.get("kind")
        feature1 = list(feature1)[0]
        if feature1 == 'peptide_bond':
            feature1_vec = [0, 1]
        else:
            feature1_vec = [1, 0]
        feature2 = [data.get("distance")]

        edge_feature = feature1_vec + feature2
        edge_features.append(edge_feature)

        # If reverse edges are being added, duplicate the feature
        edge_features.append(edge_feature)

    # Convert list of edge features to a tensor and add to DGL graph
    edge_feature_tensor = torch.tensor(edge_features)
    g_dgl.edata['e'] = edge_feature_tensor

    g_dgl = dgl.add_self_loop(g_dgl)
    return g_dgl



def create_prot_esm_embedding(pdb_name):
    from graphein.protein.features.sequence.embeddings import esm_sequence_embedding
    processing_funcs = [deprotonate_structure, convert_structure_to_centroids, remove_insertions]

    # Read dataframe from PDB
    raw_df = read_pdb_to_dataframe(path="data/DAVIS/pdb/"+pdb_name+".pdb")

    # Apply processing functions
    df = process_dataframe(raw_df, atom_df_processing_funcs=processing_funcs)

    from graphein.protein.graphs import initialise_graph_with_metadata, add_nodes_to_graph
    from graphein.protein.graphs import annotate_node_metadata, annotate_graph_metadata, annotate_edge_metadata
    g = initialise_graph_with_metadata(protein_df=df,  # from above cell
                                       raw_pdb_df=raw_df,  # Store this for traceability
                                       pdb_code=pdb_name,  # and again
                                       granularity="centroid"  # Store this so we know what kind of graph we have
                                       )
    g = add_nodes_to_graph(g)
    from graphein.protein.graphs import compute_edges
    g = compute_edges(g, get_contacts_config=None, funcs=[add_peptide_bonds, add_hydrogen_bond_interactions])
    g = annotate_node_metadata(g, [amino_acid_one_hot])
    g = annotate_graph_metadata(g, [esm_sequence_embedding])
    return g["esm_embedding_A"]
