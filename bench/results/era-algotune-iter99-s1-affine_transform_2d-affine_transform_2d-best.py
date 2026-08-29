import numpy as np
from scipy.ndimage import affine_transform

def solve(problem):
    image = np.asarray(problem["image"], dtype=np.float64)
    matrix = np.asarray(problem["matrix"], dtype=np.float64)
    n = image.shape[0]

    A = matrix[:2, :2]
    b = matrix[:2, 2]

    transformed = affine_transform(
        image,
        A,
        offset=b,
        output_shape=(n, n),
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )

    return {"transformed_image": np.ascontiguousarray(transformed, dtype=np.float64)}
