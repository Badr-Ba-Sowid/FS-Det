
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict


def save_support_query_samples(data: List[Dict[str, NDArray]], filename: str):
    output_array = np.stack(data, axis=0) # type: ignore

    np.save(filename, output_array)
