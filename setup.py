from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='dream_olfaction',  # The name of your package
    version='0.1.0',  # The initial release version
    packages=find_packages(),  # Automatically find packages in your project
    include_package_data=True,  # Include other files listed in MANIFEST.in

    # List of dependencies
    install_requires=[
        'pandas',
        'seaborn',
        'matplotlib'
    ],

    # Metadata about your package
    author='Your Name',  # Your name
    author_email='your.email@example.com',  # Your email
    description='A package for exploring and analyzing data for the DREAM Olfaction Challenge',  # Short description
    long_description=long_description,  # Long description read from the README file
    long_description_content_type='text/markdown',  # The format of the long description (Markdown)
    url='https://github.com/yourusername/dream_olfaction',  # URL of your project (e.g., GitHub repository)

    # Classifiers help users find your project by categorizing it
    classifiers=[
        'Development Status :: 3 - Alpha',  # Development status
        'Intended Audience :: Developers',  # Intended audience
        'Topic :: Software Development :: Libraries',  # Project topic
        'License :: OSI Approved :: MIT License',  # License
        'Programming Language :: Python :: 3.8',  # Supported Python versions
        'Programming Language :: Python :: 3.9',
    ],

    # Entry points specify what scripts should be made available to the command line
    entry_points={
        'console_scripts': [
            'explore=explore:main',
        ],
    },
)

