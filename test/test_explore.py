import pytest
import pandas as pd
from tonic import explore
import matplotlib

# set plot visible to false
matplotlib.use('Agg') # No display

# Sample DataFrame for testing
sample_data = {
    'IsomericSMILES': [
        'CC(C)CC(C)(O)C1CCCS1', 
        'CCCCCCCCC(=O)C(=O)OC(=C(C)C)C(O)CO', 
        'CCCCC=COC(=O)C=CCCCCC', 
        'CCCC(CCOC)[SH]=C(O)OCC', 
        'CCCCCCCCC(S)C=O'
    ],
    'floral': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(sample_data).set_index('IsomericSMILES')

def test_plot_floral_distribution():
    explore.plot_floral_distribution(df, kind='count')
    assert True

    explore.plot_floral_distribution(df, kind='pie')
    assert True 

    with pytest.raises(ValueError):
        explore.plot_floral_distribution(df, kind='invalid_kind')

def test_plot_molecular_structures():
    explore.plot_molecular_structures(df, num_samples=3)
    assert True 

def test_plot_molecular_structures_w_label():
    explore.plot_molecular_structures_w_label(df, num_samples=3)
    assert True

# Run the tests
if __name__ == "__main__":
    pytest.main()

