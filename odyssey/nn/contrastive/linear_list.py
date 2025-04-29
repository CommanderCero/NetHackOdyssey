import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Straight up ripped from pytorchs code and modified
# https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py#L50
class LinearList(nn.Module):
    """
    An indexable list of linear transformations applying an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
    Compared to `torch.nn.Linear`, this module allows for selecting different linear transformations to different inputs in a batch.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_linears: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((num_linears, out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_linears, out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
        weights = self.weight[indices]
        out = torch.bmm(weights, input.unsqueeze(-1)).squeeze(-1)
        if self.bias is None:
            return out
        return out + self.bias[indices]

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
    
if __name__ == "__main__":
    # Example usage
    batch_size = 10000

    linear_list = LinearList(100, 100, 10)
    input_tensor = torch.randn(batch_size, 100)
    indices = torch.randint(0, 10, (batch_size,))
    output = linear_list(input_tensor, indices)

    # Verify (atol=1e-7 because default atol=1e-8 is too sensitive, we rarely have some tiny numerical differences)
    for row in range(batch_size):
        assert torch.allclose(output[row], F.linear(input_tensor[row], linear_list.weight[indices[row]], bias=linear_list.bias[indices[row]]), atol=1e-7), f"Mismatch at row {row}"

    # Verify using linear
    for row in range(batch_size):
        linear = nn.Linear(10, 5)
        linear.weight = nn.Parameter(linear_list.weight[indices[row]])
        linear.bias = nn.Parameter(linear_list.bias[indices[row]]) if linear_list.bias is not None else None
        linear_output = linear(input_tensor[row])
        assert torch.allclose(output[row], linear_output, atol=1e-7), f"Mismatch at row {row} with linear layer"