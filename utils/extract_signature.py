from .find_radius import find_radius
from .preprocess_pipeline import preprocess_cone_tip
from .get_signature import get_statistical_signature


def extract_signature_from_image(bgr_image, inner_crop_pct=0.10, outer_crop_pct=0.10,
                                  bilateral_d=9, bilateral_sigma_color=75,
                                  bilateral_sigma_space=75):
    """
    Extract statistical signature from a single cone image.

    Args:
        bgr_image: Input BGR image
        inner_crop_pct: Inner crop percentage for polar warp
        outer_crop_pct: Outer crop percentage for polar warp
        bilateral_*: Bilateral filter parameters

    Returns:
        Signature dict or None if extraction failed
    """
    # Find cone and get center/radius
    cropped_img, center, radius = find_radius(bgr_image)

    if cropped_img is None:
        return None

    # Preprocess (filter -> LAB -> polar warp -> crop)
    lab_patch = preprocess_cone_tip(
        cropped_img, center, radius,
        inner_crop_pct=inner_crop_pct,
        outer_crop_pct=outer_crop_pct,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space
    )

    # Extract signature
    sig = get_statistical_signature(lab_patch)

    return sig
