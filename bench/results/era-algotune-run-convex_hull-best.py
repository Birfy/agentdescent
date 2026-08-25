import numpy as np
from scipy.spatial import ConvexHull

def solve(problem):
    points = problem['points']
    
    if isinstance(points, np.ndarray):
        if points.dtype == np.float64 and points.flags['C_CONTIGUOUS']:
            pts = points
        else:
            pts = np.ascontiguousarray(points, dtype=np.float64)
    else:
        pts = np.array(points, dtype=np.float64)
    
    n = pts.shape[0]
    if n >= 4:
        hull = ConvexHull(pts, qhull_options='QJ')
        vertices = hull.vertices
    else:
        vertices = np.arange(n)
    
    return {
        'hull_vertices': vertices.tolist(),
        'hull_points': pts[vertices].tolist()
    }
