from gnn_mesh_simplification.losses import SurfaceDistanceLoss


def test_surface_distance_loss():
    SurfaceDistanceLoss(k=5, num_points=10)
