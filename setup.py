from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='gin',  # The name of your package
    version='0.1.0',  # The initial release version
    packages=find_packages(),  # Automatically find packages in your project
    include_package_data=True,  # Include other files listed in MANIFEST.in

    # List of dependencies
    install_requires=[
        # Visualization dependencies
        'seaborn',
        'matplotlib',
        'Pillow',  # For image manipulation
        # Data manipulation dependencies
        'pandas',
        'numpy',
        # Machine learning dependencies
        'torch',
        'torch_geometric',
        'tensorboard',
        #'torch_scatter',
        # 'scikit-learn==1.2.2',
        'scikit-learn',
        'imbalanced-learn', 
        # Conversion dependencies
        'jupytext', # for conversion from .py to .ipynb
        'nbconvert', # for conversion from .py to .ipynb
        # Network dependencies
        'requests',
        # Molecular visualization dependencies
        'rdkit',
        # Testing dependencies
        'pytest',
        # Utility dependencies
        'tqdm',
        # Cheminformatics dependencies
        'rdkit',
    ],

    # Metadata about your package
    author='Ryan Young',  
    author_email='mlsci@ryanyoung.io',  
    description='A package for exploring and analyzing DREAM and Pyrfume data',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/synapticsage/tonic',  

    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers', 
        'Topic :: Software Development :: Libraries',  
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.9',  
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Smell',
        'Odor',
        'Machine Learning'
    ],
    

    # Entry points specify what scripts should be made available to the command line
    entry_points={
        'console_scripts': [
            'explore=explore:main',
        ],
    },
)

