import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from PIL import ImageDraw
from PIL import Image

import io

# Rename the classes for better visualization
rename = {0: 'Non-floral', 1: 'Floral'}

def plot_floral_distribution(df: pd.DataFrame,
                             kind: str = 'pie'):
    """Visualize the distribution of the floral."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    if kind == 'count':
        sns.countplot(x='floral', data=df, palette="viridis")
    elif kind == 'pie':
        (df['floral'].
            apply(lambda x: rename[x]).
            value_counts().plot.pie(autopct='%1.1f%%'))
    else:
        raise ValueError(f"kind must be 'count' or 'pie', not {kind}")
    plt.title('Distribution of Floral Labels', fontsize=16)
    plt.xlabel('Floral')
    plt.ylabel('Count')
    plt.show()

def plot_molecular_structures(df: pd.DataFrame, num_samples: int = 10):
    """Plot sample molecular structures from the dataset."""
    sample_smiles = df.sample(num_samples).index.tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in sample_smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
    plt.imshow(img)
    plt.axis('off')  # Hide the axes


def plot_molecular_structures_w_label(df: pd.DataFrame, num_samples: int = 10,
                                      hit_background_color: str = 'green',
                                      hit_font_color: str = 'white'):
    """
    Plot sample molecular structures from the dataset, coloring floral label hits.
    """
    sample_smiles = df.sample(num_samples).index.tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in sample_smiles]
    legends = ['Floral' if df.loc[smile, 'floral'] == 1 else 'Non-floral' for smile in sample_smiles]
    
    # Draw molecules to a grid image
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=legends)

    # Convert IPython.core.display.Image to PIL Image if necessary
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        # Convert IPython Image to PIL Image
        img_data = img.data
        pil_img = Image.open(io.BytesIO(img_data))

    # Create an ImageDraw object to highlight floral molecules
    draw = ImageDraw.Draw(pil_img)
    
    # Highlight floral molecules
    for idx, smile in enumerate(sample_smiles):
        x = (idx % 5) * 200
        y = (idx // 5) * 200
        if df.loc[smile, 'floral'] == 1:
            draw.rectangle([x, y, x+200, y+200], outline=hit_background_color, width=5)
            draw.text((x + 5, y + 5), 'Floral', fill=hit_font_color)
        # Add SMILES string as a label
        mol = mols[idx]
        mol_name = Chem.MolToSmiles(mol)
        # wrap text with 40 characters
        mol_name = '\n'.join([mol_name[i:i+40] for i in range(0, len(mol_name), 40)])
        draw.text((x + 15, y+15), mol_name, fill='black')
    
    # Display the image using IPython display
    # display(pil_img)

    # Convert PIL Image to NumPy array for matplotlib
    img_array = np.array(pil_img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis('off')  # Hide the axes
    plt.show()



def plot_feature_heatmap(x: np.ndarray, 
                         y: np.ndarray|None = None, 
                         show_labels: bool = True):
    """Plot a heatmap of the feature vectors and optionally add vertical lines for floral vectors."""
    if y is None: show_labels = False
    plt.figure(figsize=(12, 6))
    plt.imshow(x, aspect='auto', cmap='viridis', interpolation='none')
    plt.xlabel('Feature Index')
    plt.ylabel('Molecule Index')
    plt.title('Feature Heatmap')
    
    if show_labels:
        for idx in range(x.shape[0]):
            if y[idx] == 1:
                plt.axhline(y=idx, color='green', linestyle='dotted', linewidth=0.5)
    
    plt.show()

# Example usage
if __name__ == "__main__":

    from tonic import data
    df = data.read_local_csv()

    # Pie chart of floral distribution
    plot_floral_distribution(df, kind='pie')

    # Plot molecular structures
    plot_molecular_structures(df, num_samples=20)

    # Plot molecular structures with labels
    plot_molecular_structures_w_label(df, num_samples=25)

