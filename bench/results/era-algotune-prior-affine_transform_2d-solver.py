import numpy as np
import scipy.ndimage

def solve(problem):
    """
    Applies a 2D affine transformation to an input image using cubic spline
    interpolation (order=3) and constant boundary mode (padding with 0).
    
    The transformation is defined by a 2x3 matrix. The output image has the
    same shape as the input image.
    
    Args:
        problem: A dictionary with keys:
            - "image": An n x n array of floats representing the input image.
            - "matrix": A 2x3 array representing the affine transformation matrix.
    
    Returns:
        A dictionary with key:
            - "transformed_image": The transformed image array of shape (n, n).
    """
    image = problem['image']
    matrix = problem['matrix']
    
    # Convert inputs to numpy arrays for efficient processing
    if not isinstance(image, np.ndarray):
        image = np.asarray(image, dtype=np.float64)
    else:
        image = np.asarray(image, dtype=np.float64)
    
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
    
    # Use scipy's affine_transform with cubic spline interpolation (order=3)
    # and constant boundary mode (padding with 0)
    transformed_image = scipy.ndimage.affine_transform(
        image, 
        matrix, 
        order=3, 
        mode='constant'
    )
    
    return {'transformed_image': transformed_image}
