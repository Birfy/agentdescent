import numpy as np
from scipy.ndimage import affine_transform

def solve(problem):
    image = problem['image']
    matrix = problem['matrix']
    # Ensure float64 and contiguous without extra copy if possible
    if image.dtype != np.float64 or not image.flags.c_contiguous:
        image = np.ascontiguousarray(image, dtype=np.float64)
    if matrix.dtype != np.float64 or not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    
    transformed = affine_transform(
        image,
        matrix,
        order=3,
        mode='constant',
        cval=0.0
    )
    
    return {'transformed_image': transformed}
