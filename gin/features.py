import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import RDKFingerprint
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# from rdkit.Chem import rdmolops

import matplotlib

# from tqdm.auto import tqdm
# from typing import List

DEFAULT_MORGAN_BITS = 1024
DEFAULT_MACCS_BITS = 167
DEFAULT_RDKIT_BITS = 2048

def get_morgan_fingerprint(smiles_str: str, radius: int = 2, 
                           n_bits: int = DEFAULT_MORGAN_BITS,
                           output: str='default'):
    """Generate a Morgan (circular) fingerprint for a molecule."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_str}")
    gen = GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    fingerprint = gen.GetFingerprint(mol)

    

    if output == 'array':
        fingerprint = np.array(fingerprint)
    DEFAULT_MORGAN_BITS = n_bits # Update the default number of bits
    return fingerprint

def get_maccs_keys_fingerprint(smiles_str: str, output: str='default'):
    """Generate a MACCS keys fingerprint for a molecule."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_str}")
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    if output == 'array':
        fingerprint = np.array(fingerprint)
    return fingerprint

def get_combined_fingerprint(smiles_str: str, radius: int = 2, n_bits: int = 1024):
    """Generate a combined Morgan and MACCS keys fingerprint for a molecule."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_str}")
    
    # Generate Morgan fingerprint
    gen = GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    morgan_fp = gen.GetFingerprint(mol)
    morgan_fp = np.array(morgan_fp)
    
    # Generate MACCS keys fingerprint
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_fp = np.array(maccs_fp)


    # Generate RDKit fingerprint
    rdkit_fp = RDKFingerprint(mol)
    rdkit_fp = np.array(rdkit_fp)
    
    # Concatenate fingerprints
    combined_fp = np.concatenate((morgan_fp, maccs_fp, rdkit_fp))
    
    return combined_fp


def plot_fingerprint(fingerprint, **kwargs):
    """Plot a fingerprint as a series of bits."""
    if not isinstance(fingerprint, np.ndarray):
        fingerprint = np.array(fingerprint)
    from matplotlib import pyplot as plt
    plt.scatter(range(len(fingerprint)), fingerprint, marker='|', **kwargs)
    plt.xlabel('Bit index')
    plt.ylabel('Fingerprint')
    plt.show()

##

def get_top_important_features_from_molecules(smiles_list, top_n=5):
    combined_fps = [get_combined_fingerprint(smiles) for smiles in smiles_list]
    combined_fp = np.sum(combined_fps, axis=0)  # Summing the fingerprints to find common features
    highest_indices = np.argsort(combined_fp)[::-1]  # Sort in descending order
    # np.random.shuffle(highest_indices)  # Shuffle the indices to get a random selection
    top_features = highest_indices[:top_n]  # Select the top N features
    return top_features

##

def highlight_features_on_molecule(smiles, top_features,
    morgan_len = DEFAULT_MORGAN_BITS, maccs_len = 167, rdkit_len = 2048):
    """
    Highlight the atoms and bonds in a molecule corresponding to the top features.

    Args:
    - smiles (str): SMILES string of the molecule.
    - top_features (List[int]): List of top feature indices to highlight.
    - morgan_len (int): Number of bits in the Morgan fingerprint.
    - maccs_len (int): Number of bits in the MACCS keys fingerprint.
    - rdkit_len (int): Number of bits in the RDKit fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    combined_len = morgan_len + maccs_len + rdkit_len

    if len(top_features) == 0:
        return Draw.MolToImage(mol, size=(300, 300))
    
    combined_fp = get_combined_fingerprint(smiles)
    
    atom_highlight = []
    bond_highlight = []
    
    for bit in top_features:
        if bit < morgan_len:
            if combined_fp[bit]:
                info = {}
                if bit in info:
                    for atom_idx, radius in info[bit]:
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                        atom_highlight.extend([mol.GetBondWithIdx(idx).GetBeginAtomIdx() for idx in env])
                        atom_highlight.extend([mol.GetBondWithIdx(idx).GetEndAtomIdx() for idx in env])
                        bond_highlight.extend(env)
        elif bit < morgan_len + maccs_len:
            bit -= morgan_len
            if combined_fp[morgan_len + bit]:
                pass  # Add logic if needed for MACCS keys
        elif bit < combined_len:
            bit -= morgan_len + maccs_len
            if combined_fp[morgan_len + maccs_len + bit]:
                pass  # Add logic if needed for RDKit fingerprint
    
    atom_highlight = list(set(atom_highlight))  # Remove duplicates
    bond_highlight = list(set(bond_highlight))  # Remove duplicates
    
    return Draw.MolToImage(mol, highlightAtoms=atom_highlight, highlightBonds=bond_highlight, size=(300, 300),
                           highlightColor=matplotlib.colors.to_rgb('purple'))

def plot_molecules_with_highlighted_features(smiles_list, 
                                             top_features=None,
                                             top_n=5):
    if top_features is None:
        top_features = get_top_important_features_from_molecules(smiles_list, top_n)
    
    for smiles in smiles_list:
        img = highlight_features_on_molecule(smiles, top_features)
        if img:
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.show()

def featurize_smiles(smiles_str: str,
                     method: str = 'combined') -> np.ndarray:
  """Convert a molecule SMILES into a 1D feature vector."""
  if method == 'morgan':
    fingerprint = tonic.features.get_morgan_fingerprint(smiles_str)
  elif method == 'maccs':
    fingerprint = tonic.features.get_maccs_keys_fingerprint(smiles_str)
  elif method == 'combined':
    fingerprint = tonic.features.get_combined_fingerprint(smiles_str)
  else:
    raise ValueError(f"Invalid method: {method}")
  return fingerprint

# Example usage
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from gin.features import get_morgan_fingerprint, get_maccs_keys_fingerprint, get_combined_fingerprint

    smiles = 'CCO'
    print("Morgan Fingerprint:", get_morgan_fingerprint(smiles))
    print("MACCS Keys Fingerprint:", get_maccs_keys_fingerprint(smiles))
    print("Combined Fingerprint:", get_combined_fingerprint(smiles))

    plot_fingerprint(get_morgan_fingerprint(smiles), label='Morgan Fingerprint')
    plot_fingerprint(get_maccs_keys_fingerprint(smiles), label='MACCS Keys')
    plt.legend()

    plot_fingerprint(get_combined_fingerprint(smiles), label='Combined Fingerprint')

    smiles = ['CCO', 'CCN', 'CCC']
    top_features = get_top_important_features_from_molecules(smiles)
    print("Top features:", top_features)

    smiles_list = ["CCO", "CCCC", "c1ccccc1"]
    top_features = get_top_important_features_from_molecules(smiles_list)
    plot_molecules_with_highlighted_features(smiles_list, top_features)
