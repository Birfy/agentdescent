import numpy as np
from scipy.ndimage import affine_transform

def solve(problem):
    image = np.asarray(problem['image'], dtype=np.float64)
    matrix = np.asarray(problem['matrix'], dtype=np.float64)
    
    # Build the full 3x3 homogeneous matrix
    M = np.eye(3)
    if matrix.shape == (2, 2):
        M[:2, :2] = matrix
    elif matrix.shape == (2, 3):
        M[:2, :] = matrix
    elif matrix.shape == (3, 3):
        M = matrix
    else:
        raise ValueError("Unsupported matrix shape")
    
    # Use the 2x3 part directly as scipy expects output->input mapping
    matrix_2x3 = M[:2, :]
    
    transformed = affine_transform(
        image,
        matrix_2x3,
        output_shape=image.shape,
        order=3,
        mode='constant',
        cval=0.0,
        prefilter=True
    )
    
    return {'transformed_image': np.ascontiguousarray(transformed, dtype=np.float64)}
