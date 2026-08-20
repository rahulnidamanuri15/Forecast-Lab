import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_db_connection():
    """Mock database connection for tests."""
    with patch('app.get_db_connection') as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
        yield mock_cursor