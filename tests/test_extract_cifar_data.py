import os

def test_data_file():
    test_filename = 'cifar.tar.gz'
    # Check if the file was created
    assert os.path.exists(test_filename), "File was not downloaded"
        
    # Clean up (remove the downloaded test file)
    if os.path.exists(test_filename):
        os.remove(test_filename)
