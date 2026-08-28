import numpy as np
from scipy.ndimage import affine_transform

def solve(problem):
    image = np.asarray(problem["image"], dtype=np.float64)
    matrix = np.asarray(problem["matrix"], dtype=np.float64)
    
    n = image.shape[0]
    
    # Matrix is 2x3: [ [a, b, tx], [c, d, ty] ]
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    
    # The matrix defines the inverse mapping: output -> input.
    # In image coordinates (row, col):
    # row_in = a*row_out + b*col_out + tx
    # col_in = c*row_out + d*col_out + ty
    F = np.array([[a, b],
                  [c, d]], dtype=np.float64)
    offset = np.array([tx, ty], dtype=np.float64)
    
    # scipy.ndimage.affine_transform expects:
    # output[o] = input[F @ o + offset]
    transformed = affine_transform(
        image,
        F,
        offset=offset,
        output_shape=(n, n),
        order=3,
        mode="constant",
        cval=0.0,
    )
    
    return {"transformed_image": np.asarray(transformed, dtype=np.float64)}
