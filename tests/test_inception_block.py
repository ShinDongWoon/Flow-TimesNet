import sys
from pathlib import Path

import pytest
import torch

# Ensure the project src is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from timesnet_forecast.models.timesnet import InceptionBlock


@pytest.mark.parametrize(
    "in_ch,out_ch,bottleneck_ratio",
    [
        (8, 8, 0.5),
        (4, 6, 2.0),
    ],
)
def test_inception_block_bottleneck_residual(in_ch, out_ch, bottleneck_ratio):
    block = InceptionBlock(
        in_ch=in_ch,
        out_ch=out_ch,
        kernel_set=[(3, 3), (5, 1)],
        dropout=0.0,
        act="gelu",
        bottleneck_ratio=bottleneck_ratio,
    )

    with torch.no_grad():
        x = torch.randn(2, in_ch, 5, 7)
        out = block(x)

    assert out.shape == (2, out_ch, 5, 7)

    with torch.no_grad():
        residual = block.res_proj(x)
        branch_outputs = [path(x) for path in block.paths]
        merged = torch.cat(branch_outputs, dim=1)
        projected = block.proj(merged)
        activated = block.act(projected)
        dropped = block.dropout(activated)

    assert torch.allclose(out, dropped + residual)
