import os
import bpy
import math
import bmesh

import mathutils
from mathutils import Vector
from mathutils import Matrix


import numpy as np
from scipy.spatial import cKDTree


#------------------------------------------------------------------#
#------------------------------------------------------------------#


#------------------------------------------------------------------#
#------------------------------------------------------------------#
def icp(source, target, max_iterations=50, tolerance=1e-5):

    def find_closest_points(source, target):
        tree = cKDTree(target)
        distances, indices = tree.query(source)
        return target[indices], distances

    def compute_transformation(src, tgt):
        src_mean = np.mean(src, axis=0)
        tgt_mean = np.mean(tgt, axis=0)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean
        return R, t

    def apply_transformation(points, R, t):
        return (R @ points.T).T + t

    # Initialize variables
    src = source.copy()
    transformation = np.eye(4)

    for i in range(max_iterations):
        closest_points, distances = find_closest_points(src, target)
        R, t = compute_transformation(src, closest_points)
        src = apply_transformation(src, R, t)

        # Update the global transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        transformation = T @ transformation

        # Check for convergence
        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    return src, transformation


#------------------------------------------------------------------#
def object_vertices_get(obj_name) :
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)    
    mesh = obj.data
    
    vertices = mesh.vertices
    vertices = [obj.matrix_world @ vertex.co for vertex in vertices]
    vertices = [[v.x, v.y, v.z] for v in vertices]
    return np.array(vertices)


#------------------------------------------------------------------#
def import_and_get_new_objects(import_operator, collection, **kwargs) :
    bpy.ops.object.select_all(action='DESELECT')

    existing_objects = set(bpy.data.objects)
    getattr(bpy.ops.wm, import_operator)(**kwargs)
    new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
    
    if collection : 
        for obj in new_objects : 
            for col in obj.users_collection: col.objects.unlink(obj)
            collection.objects.link(obj)
        
    return new_objects


def clean_all_objects():
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False)
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#
## 0.0.
clean_all_objects()


filepath = r"F:\Models\Blender\object.ply"
source = import_and_get_new_objects("ply_import", None, filepath=filepath)
source = bpy.data.objects.get("object")  
source = object_vertices_get("object")
print(source)


filepath = r"F:\Models\Blender\reference.ply"
target = import_and_get_new_objects("ply_import", None, filepath=filepath)
target = bpy.data.objects.get("reference")  
target = object_vertices_get("reference")
print(target)


aligned_source, final_transformation = icp(source, target)
print("Final Transformation Matrix:")
print(final_transformation)


obj = bpy.data.objects.get("object")  
obj.matrix_world = mathutils.Matrix(final_transformation.tolist())
obj.select_set(True) 


#------------------------------------------------------------------#
#------------------------------------------------------------------#