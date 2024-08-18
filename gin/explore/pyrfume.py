import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import ImageDraw, Image
import io

# Rename the classes for better visualization
def rename(descriptor: str = 'floral') -> dict:
    return {0: f'Non-{descriptor}', 1: f'{descriptor.capitalize()}'}

def plot_desc_distribution(df: pd.DataFrame, descriptor: str = 'floral', kind: str = 'pie'):
    """Visualize the distribution of a descriptor."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    if kind == 'count':
        sns.countplot(x=descriptor, data=df, palette="viridis")
    elif kind == 'pie':
        (df[descriptor].
            apply(lambda x: rename(descriptor)[x]).
            value_counts().plot.pie(autopct='%1.1f%%'))
    else:
        raise ValueError(f"kind must be 'count' or 'pie', not {kind}")
    plt.title(f'Distribution of {descriptor.capitalize()} Labels', fontsize=16)
    plt.xlabel(descriptor.capitalize())
    plt.ylabel('Count')
    plt.show()

def plot_molecular_structures(df: pd.DataFrame, num_samples: int = 10):
    """Plot sample molecular structures from the dataset."""
    sample_smiles = df.sample(num_samples).index.tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in sample_smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()

def plot_molecular_structures_w_label(df: pd.DataFrame, descriptor: str = 'floral',
                                      num_samples: int = 10, hit_background_color: str = 'green',
                                      hit_font_color: str = 'white'):
    """
    Plot sample molecular structures from the dataset, coloring descriptor label hits.
    """
    sample_smiles = df.sample(num_samples).index.tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in sample_smiles]
    legends = [f'{descriptor.capitalize()}' if df.loc[smile, descriptor] == 1 else f'Non-{descriptor}' for smile in sample_smiles]
    
    # Draw molecules to a grid image
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=legends)

    # Convert IPython.core.display.Image to PIL Image if necessary
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        # Convert IPython Image to PIL Image
        img_data = img.data
        pil_img = Image.open(io.BytesIO(img_data))

    # Create an ImageDraw object to highlight descriptor molecules
    draw = ImageDraw.Draw(pil_img)
    
    # Highlight molecules with the descriptor
    for idx, smile in enumerate(sample_smiles):
        x = (idx % 5) * 200
        y = (idx // 5) * 200
        if df.loc[smile, descriptor] == 1:
            draw.rectangle([x, y, x+200, y+200], outline=hit_background_color, width=5)
            draw.text((x + 5, y + 5), f'{descriptor.capitalize()}', fill=hit_font_color)
        # Add SMILES string as a label
        mol = mols[idx]
        mol_name = Chem.MolToSmiles(mol)
        mol_name = '\n'.join([mol_name[i:i+40] for i in range(0, len(mol_name), 40)])  # Wrap text with 40 characters
        draw.text((x + 15, y + 15), mol_name, fill='black')
    
    # Convert PIL Image to NumPy array for matplotlib
    img_array = np.array(pil_img)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis('off')  # Hide the axes
    plt.show()

def plot_feature_heatmap(x: np.ndarray, y: np.ndarray|None = None, show_labels: bool = True):
    """Plot a heatmap of the feature vectors and optionally add vertical lines for descriptor vectors."""
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor', type=str, default='floral', help='Descriptor to visualize')
    args = parser.parse_args()

    descriptor = args.descriptor  # or any other descriptor like 'sweet', 'bitter', etc.

    from gin.data.pyrfume import get_join
    df = get_join('leffingwell')
    df = df.set_index('SMILES')
    df = df[[descriptor]]


    # Pie chart of descriptor distribution
    plot_desc_distribution(df, descriptor=descriptor, kind='pie')

    # Plot molecular structures
    plt.figure(figsize=(12, 6))
    plot_molecular_structures(df, num_samples=20)

    # Plot molecular structures with labels
    plt.figure(figsize=(12, 6))
    plot_molecular_structures_w_label(df, descriptor=descriptor, num_samples=25)
