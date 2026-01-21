import numpy as np
import json


def save_template(template, output_path):
    """
    Save template to disk (numpy for histogram, JSON for metadata).

    Args:
        template: Template dict from train_template()
        output_path: Base path for output files (without extension)
    """
    # Save histogram as numpy file
    np.save(f"{output_path}_hist.npy", template['histogram'])

    # Prepare JSON-serializable version
    template_json = template.copy()
    template_json['histogram'] = f"{output_path}_hist.npy"

    # Save metadata as JSON
    with open(f"{output_path}_meta.json", 'w') as f:
        json.dump(template_json, f, indent=2)

    print(f"Template saved: {output_path}_hist.npy, {output_path}_meta.json")
