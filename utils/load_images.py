import cv2
import os


def load_images_from_directory(directory, extensions=('.png', '.jpg', '.jpeg')):
    """
    Load all images from a directory.

    Args:
        directory: Path to directory containing images
        extensions: Tuple of valid file extensions

    Returns:
        List of BGR images
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(extensions):
            path = os.path.join(directory, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images
