import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's and torch's random generators (CPU and every GPU) with `seed`.

    The one place a run sets its randomness; call it once
    at the start of each pass that draws random numbers.

    Parameters
    ----------
    seed : int
        The seed from the study's configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
