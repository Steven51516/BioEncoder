import re
import numpy as np

AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

def CalculateDipeptideComposition(ProteinSequence):
    """
    Calculate the composition of dipeptidefor a given protein sequence.
    Usage:
    result=CalculateDipeptideComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    400 dipeptides.
    """

    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round(
                float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1) * 100, 2
            )
    return Result


def GetSpectrumDict(proteinsequence):
    """
    ########################################################################
    Calcualte the spectrum descriptors of 3-mers for a given protein.
    Usage:
    result=GetSpectrumDict(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition values of 8000
    3-mers.
    """
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence))
    return result


def Getkmers():
    """
    ########################################################################
    Get the amino acid list of 3-mers.
    Usage:
    result=Getkmers()
    Output: result is a list form containing 8000 tri-peptides.
    ########################################################################
    """
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers

def CalculateAAComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of Amino acids
    for a given protein sequence.
    Usage:
    result=CalculateAAComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing the composition of
    20 amino acids.
    ########################################################################
    """
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result


def CalculateAADipeptideComposition(ProteinSequence):
    """
    ########################################################################
    Calculate the composition of AADs, dipeptide and 3-mers for a
    given protein sequence.
    Usage:
    result=CalculateAADipeptideComposition(protein)
    Input: protein is a pure protein sequence.
    Output: result is a dict form containing all composition values of
    AADs, dipeptide and 3-mers (8420).
    ########################################################################
    """

    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))
    result.update(GetSpectrumDict(ProteinSequence))

    return np.array(list(result.values()))


