import numpy as np
import json


def load_template(base_path):
    """
    Load template from disk.

    Args:
        base_path: Base path used when saving (without extension)

    Returns:
        Template dict with histogram as numpy array
    """
    # Load metadata
    with open(f"{base_path}_meta.json", 'r') as f:
        template = json.load(f)

    # Load histogram
    template['histogram'] = np.load(f"{base_path}_hist.npy")

    return template
