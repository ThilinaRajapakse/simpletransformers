from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
# adapted from:
# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/tversky.html


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N,)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: int = 0,
        scale: float = 1.0,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
        smooth: float = 0,
    ) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: int = gamma
        self.scale: float = scale
        self.reduction: Optional[str] = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps
        self.smooth: float = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if len(input.shape) == 2:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "number of elements in input and target shapes must be the same. Got: {}".format(
                        input.shape, input.shape
                    )
                )
        else:
            raise ValueError(
                "Invalid input shape, we expect or NxC. Got: {}".format(input.shape)
            )
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        input_soft = self.scale * ((1 - input_soft) ** self.gamma) * input_soft

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index

        target = target[input_mask]
        input_soft = input_soft[input_mask]

        # create the labels one hot tensor
        target_one_hot = (
            F.one_hot(target, num_classes=input.shape[1])
            .to(input.device)
            .type(input_soft.dtype)
        )

        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, -1)
        fps = torch.sum(input_soft * (1.0 - target_one_hot), -1)
        fns = torch.sum((1.0 - input_soft) * target_one_hot, -1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (numerator + self.smooth) / (
            denominator + self.eps + self.smooth
        )
        tversky_loss = 1.0 - tversky_loss

        if self.reduction is None or self.reduction == "none":
            return tversky_loss
        elif self.reduction == "mean":
            return torch.mean(tversky_loss)
        elif self.reduction == "sum":
            return torch.sum(tversky_loss)
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
