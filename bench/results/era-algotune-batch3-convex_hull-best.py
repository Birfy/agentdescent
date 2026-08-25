import numpy as np
from scipy.spatial import ConvexHull

def solve(problem):
    points = problem['points']
    
    if isinstance(points, np.ndarray):
        pts = points
    else:
        pts = np.asarray(points, dtype=np.float64)

    hull = ConvexHull(pts)
    vertices = hull.vertices
    hull_points = pts[vertices]
    
    return {
        'hull_vertices': vertices.tolist(),
        'hull_points': hull_points.tolist()
    }
