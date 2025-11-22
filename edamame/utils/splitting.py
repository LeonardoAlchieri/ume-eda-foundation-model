import itertools
import math
import warnings
from typing import TypeVar, Optional, Generator, Union
from collections.abc import Sequence

_T = TypeVar("_T")

from torch.utils.data import Subset, Dataset
from torch import randperm, default_generator, cat
from pandas import DataFrame


def user_session_splitting(
    dataset: Dataset[_T],
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> list[Subset[_T]]:

    if not hasattr(dataset, "series_to_idx"):
        raise ValueError(
            "Dataset must have 'series_to_idx' attribute for user_session_splitting."
        )

    if math.isclose(sum(lengths), 1.0) and sum(lengths) <= 1:
        lengths = [math.floor(len(dataset.series_to_idx) * l) for l in lengths]
        # Adjust the last length to match the total length
        lengths[-1] += len(dataset.series_to_idx) - sum(lengths)
    elif sum(lengths) > 1:
        raise NotImplementedError("Lengths summing to more than 1 are not supported.")
    else:
        raise ValueError(
            f"Received lengths that do not sum to 1 or are not fractions. Got {lengths}"
        )

    series_indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]

    return [
        Subset(
            dataset,
            cat(
                [
                    
                    dataset.series_to_idx[i]
                    for i in series_indices[offset - length : offset]
                ]
            ),
        )
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
