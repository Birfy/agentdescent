import numpy as np
from scipy.ndimage import affine_transform

def solve(problem):
    image = np.ascontiguousarray(problem['image'], dtype=np.float64)
    matrix = np.ascontiguousarray(problem['matrix'], dtype=np.float64)
    output = np.empty_like(image)
    affine_transform(image, matrix, output=output, order=3, mode='constant', cval=0.0)
    return {'transformed_image': output}
