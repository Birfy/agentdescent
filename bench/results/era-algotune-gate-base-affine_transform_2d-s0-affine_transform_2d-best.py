import numpy as np
import scipy.ndimage

def solve(problem):
    image = np.asarray(problem['image'], dtype=np.float64)
    matrix = np.asarray(problem['matrix'], dtype=np.float64)
    
    transformed = scipy.ndimage.affine_transform(
        image,
        matrix,
        order=3,
        mode='constant',
        cval=0.0,
        output=np.empty_like(image)
    )
    return {'transformed_image': transformed}
