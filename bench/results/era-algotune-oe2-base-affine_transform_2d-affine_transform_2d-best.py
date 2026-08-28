import numpy as np
import scipy.ndimage

def solve(problem):
    image = problem['image']
    matrix = problem['matrix']
    
    transformed = scipy.ndimage.affine_transform(
        image,
        matrix,
        order=3,
        mode='constant',
        cval=0.0,
        prefilter=True
    )
    
    return {'transformed_image': transformed}
