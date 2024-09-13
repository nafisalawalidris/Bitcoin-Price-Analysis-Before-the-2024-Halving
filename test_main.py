import pytest
import pandas as pd
import tempfile
import os
from main import preprocess_data, load_data

@pytest.fixture
def temp_csv_file():
    # Create a temporary CSV file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_file.write(b'Date,Close\n2023-01-01,100\n2023-01-02,105\n')
    tmp_file.close()  # Close the file so it can be read by pandas
    yield tmp_file.name  # Provide the file path to the test
    os.remove(tmp_file.name)  # Clean up the file after the test

def test_load_data(temp_csv_file):
    df = load_data(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
