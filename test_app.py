import pytest
from app import app 

@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    with app.test_client() as client:
        yield client


def test_home():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert b"Predict and Prevent Customer Churn" in response.data

def test_predictdata():
    response = app.test_client().get('/predictdata')
    assert response.status_code == 200
    assert b"Enter Customer Data" in response.data