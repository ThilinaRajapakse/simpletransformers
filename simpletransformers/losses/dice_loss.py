from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
# adapted from:
# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N,)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(2, N, requires_grad=True)
        >>> target = torch.empty(2, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        gamma: int = 0,
        scale: float = 1.0,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
        smooth: float = 0,
    ) -> None:
        super(DiceLoss, self).__init__()
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
            F.one_hot(target, num_classes=input_soft.shape[-1])
            .to(input.device)
            .type(input_soft.dtype)
        )

        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, dim=-1)
        cardinality = torch.sum(input_soft + target_one_hot, dim=-1)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.eps + self.smooth
        )
        dice_loss = 1.0 - dice_score

        if self.reduction is None or self.reduction == "none":
            return dice_loss
        elif self.reduction == "mean":
            return torch.mean(dice_loss)
        elif self.reduction == "sum":
            return torch.sum(dice_loss)
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
