template_alignment.cpp : Transformation that aligns the given point cloud of object to the point cloud of current scene to find object's orientation and position.  The initial alignment is based on features and further refinement is done using Iterative Closest Point (ICP) Algorithm.

planar_segmentation.cpp : 
- Extracts the point cloud of ground plane
- Gets the normal vector to plane
- Aligns scene's point cloud such that the ground plane's normal vector is in the direction assigned to gravity (y = -1)

