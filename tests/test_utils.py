"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from utils import check_dir_exists, print_arr_id, get_file_name


class TestCheckDirExists:
    """Test directory checking/creation utility."""
    
    def test_existing_directory(self, tmp_path):
        """Test with an existing directory."""
        test_dir = tmp_path / "existing_dir"
        test_dir.mkdir()
        
        result = check_dir_exists(str(test_dir))
        assert result == str(test_dir)
        assert test_dir.exists()
    
    def test_nonexistent_directory(self, tmp_path):
        """Test creating a new directory."""
        test_dir = tmp_path / "new_dir"
        assert not test_dir.exists()
        
        result = check_dir_exists(str(test_dir))
        assert result == str(test_dir)
        assert test_dir.exists()


class TestPrintArrId:
    """Test array identification printing utility."""
    
    def test_numpy_array_info(self, capsys):
        """Test printing numpy array information."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        print_arr_id(arr, arr_name="test_array", numpyCreature=1)
        
        captured = capsys.readouterr()
        assert "test_array" in captured.out
        assert "dimensions: 2" in captured.out
        assert "shape: (2, 3)" in captured.out
        assert "size: 6" in captured.out
    
    def test_array_with_zeros(self, capsys):
        """Test array with zeros."""
        arr = np.array([[0, 1, 0], [0, 0, 2]])
        print_arr_id(arr, arr_name="sparse_array", numpyCreature=1)
        
        captured = capsys.readouterr()
        assert "nonzeros: 2" in captured.out
    
    def test_empty_array(self, capsys):
        """Test with empty array."""
        arr = np.array([])
        print_arr_id(arr, arr_name="empty", numpyCreature=1)
        
        captured = capsys.readouterr()
        assert "empty" in captured.out


class TestGetFileName:
    """Test filename extraction utility."""
    
    def test_unix_path(self):
        """Test Unix-style path."""
        path = "/home/user/data/file.txt"
        result = get_file_name(path, suffix=".txt")
        assert result == "file"
    
    def test_windows_path(self):
        """Test Windows-style path."""
        path = "C:\\Users\\data\\file.csv"
        result = get_file_name(path, suffix=".csv")
        assert result == "file"
    
    def test_no_suffix(self):
        """Test without suffix parameter."""
        path = "/path/to/myfile.h5"
        result = get_file_name(path, suffix=None)
        assert result == "myfile.h5"
    
    def test_path_without_directory(self):
        """Test filename without directory."""
        path = "simple_file.npy"
        result = get_file_name(path, suffix=".npy")
        assert result == "simple_file"
    
    def test_nested_path(self):
        """Test deeply nested path."""
        path = "/deep/nested/path/structure/data.pkl"
        result = get_file_name(path, suffix=".pkl")
        assert result == "data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
