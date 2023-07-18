import torch

from torch import nn
from torch.autograd import Function

from typing import List
from typing import TypeVar

Tensor = TypeVar('torch.tensor')

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGrad.apply


class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, dop: float = 0.1, act_fn=nn.SELU, out_fn=None, gr_flag=False, **kwargs) -> None:
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        if gr_flag:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn()
            )



    def forward(self, input: Tensor) -> Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)

        return output
