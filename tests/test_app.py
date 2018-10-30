import json
import pytest

import app


@pytest.fixture
def client():
    client = app.app.test_client()
    yield client


def test_match(client):
    res = client.post(
        '/match',
        data={
            'file': (open('tests/test.jpg', 'rb'), 'test.jpg')
        }
    )

    assert res.status_code == 200

    # data = json.loads(res.data)
    # print('data:', data)
    # assert len(data) > 0


def test_no_file(client):
    pass
