from crafter.core import Crafter
from crafter.test.resources import res
import pytest
import numpy as np

@pytest.fixture
def crafter():
    return Crafter()

def test_smoke():
    # review /tmp/crafter for the rendered results
    crafter = Crafter(output_dir='/tmp/crafter', export_extra=True)

    result = crafter.detect_text(res('idcard2.jpg'))
    assert result['boxes'].shape == (22, 4, 2)
    boxes = result['boxes']
    np.testing.assert_allclose(boxes[0], np.array([
        [692.1875, 43.75],
        [937.5, 43.75],
        [937.5, 68.75],
        [692.1875, 68.75]
    ]))


def test_size_340():
    crafter = Crafter(output_dir='/tmp/crafter-340', export_extra=True, long_size=340)
    result = crafter.detect_text(res('idcard2.jpg'))
    assert result['boxes'].shape == (14, 4, 2)
    boxes = result['boxes']
    np.testing.assert_allclose(boxes[0], np.array([
        [176.47058 ,  35.294117],
        [576.4706  ,  35.294117],
        [576.4706  , 100.      ],
        [176.47058 , 100.      ]
    ]))
