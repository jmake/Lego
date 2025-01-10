import os
import bpy
import math
import mathutils

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

import scipy 
import scipy.optimize

#------------------------------------------------------------------#
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
    

#------------------------------------------------------------------#
def save_vertices_as_ply(obj_name, file_path):
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found or is not a mesh.")
        return False
    
    # Ensure the object has mesh data
    mesh = obj.data
    vertices = mesh.vertices

    try:
        # Write PLY file
        with open(file_path, 'w') as ply_file:
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write(f"element vertex {len(vertices)}\n")
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            ply_file.write("end_header\n")
            
            for vertex in vertices:
                co = obj.matrix_world @ vertex.co  # Transform to world coordinates
                ply_file.write(f"{co.x} {co.y} {co.z}\n")
        
        print(f"Vertices saved to '{file_path}'")
        return True
    except Exception as e:
        print(f"Failed to save PLY: {e}")
        return False


#------------------------------------------------------------------#
def save_object_as_glb(obj_name, file_path):
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return False

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    
    temp_collection = bpy.data.collections.new("TempExportCollection")
    bpy.context.scene.collection.children.link(temp_collection)
    temp_collection.objects.link(obj)

    try:
        bpy.ops.export_scene.gltf(filepath=file_path, 
            use_selection=True, 
            export_apply=True)
        print(f"Object '{obj_name}' saved as .glb to '{file_path}'.")
    except Exception as e:
        print(f"Failed to save as .glb: {e}")
        return False
    finally:
        temp_collection.objects.unlink(obj)
        bpy.data.collections.remove(temp_collection)
        
        obj.select_set(False) 
        bpy.context.view_layer.objects.active = None

    return True


#------------------------------------------------------------------#
def object_apply(obj_name, func) : 
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return False

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    func(obj, func) 

    obj.select_set(False) 
    bpy.context.view_layer.objects.active = None


#------------------------------------------------------------------#
def object_rotate(obj_name, angle_degrees, orient_axis='Z'):
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return False

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    angle_radians = math.radians(angle_degrees)
    bpy.ops.transform.rotate(value=angle_radians, orient_axis=orient_axis)

    obj.select_set(False) 
    bpy.context.view_layer.objects.active = None




#------------------------------------------------------------------#
def object_origin_set(obj_name) : 
    bpy.ops.object.select_all(action='DESELECT')

    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print(f"Object '{obj_name}' not found.")
        return False

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

    obj.select_set(False) 
    bpy.context.view_layer.objects.active = None


#------------------------------------------------------------------#
def clean_all_objects():
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False)
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


#------------------------------------------------------------------#
def object_origin_set(obj) : 
    bpy.ops.object.select_all(action='DESELECT')

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ##bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')

    obj.select_set(False) 
    bpy.context.view_layer.objects.active = None


#------------------------------------------------------------------#
def object_vertices_get(obj) :
    bpy.ops.object.select_all(action='DESELECT')

    mesh = obj.data    
    vertices = mesh.vertices
    vertices = [obj.matrix_world @ vertex.co for vertex in vertices]
    vertices = [[v.x, v.y, v.z] for v in vertices]
    return np.array(vertices)


#------------------------------------------------------------------#
import numpy as np
from scipy.spatial import cKDTree

def find_closest_points_no_duplicates_iterative(source, target):
    """
    Finds the closest points in `target` for each point in `source`,
    ensuring there are no duplicate indices in the result.
    """
    tree = cKDTree(target)
    assigned_indices = set()  # Tracks already assigned indices
    final_indices = np.full(len(source), -1)  # Initialize with -1 (unassigned)
    final_distances = np.full(len(source), np.inf)  # Initialize with infinity

    remaining_source = source  # Points still to be assigned
    remaining_indices = np.arange(len(source))  # Indices of remaining points

    while len(remaining_source) > 0:
        # Query the KDTree for the closest target points to remaining_source
        distances, indices = tree.query(remaining_source)

        # Resolve duplicates
        used_indices = {}
        unassigned = []

        for i, (distance, index) in enumerate(zip(distances, indices)):
            source_idx = remaining_indices[i]

            if index not in assigned_indices:
                # If target index is unused, assign it
                final_indices[source_idx] = index
                final_distances[source_idx] = distance
                assigned_indices.add(index)
            else:
                # If target index is already used, check if it's closer or defer
                if index in used_indices:
                    # Compare distances for the conflict
                    existing_source_idx = used_indices[index]
                    if distance < final_distances[existing_source_idx]:
                        # Reassign the closer point
                        unassigned.append(existing_source_idx)
                        final_indices[existing_source_idx] = -1
                        final_distances[existing_source_idx] = np.inf

                        used_indices[index] = source_idx
                        final_indices[source_idx] = index
                        final_distances[source_idx] = distance
                    else:
                        # Keep the existing assignment and defer the current point
                        unassigned.append(source_idx)
                else:
                    # First conflict, assign and track
                    used_indices[index] = source_idx
                    final_indices[source_idx] = index
                    final_distances[source_idx] = distance

        # Prepare for next iteration: collect unassigned points
        if len(unassigned) == 0:
            break  # All points are assigned, exit the loop

        remaining_indices = np.array(unassigned)
        remaining_source = source[remaining_indices]

    # Return the unique closest points and their distances
    return target[final_indices], final_distances



#------------------------------------------------------------------#
def icp1(source, target, max_iterations=100, tolerance=1e-4):

    def find_closest_points(source, target):
        tree = cKDTree(target)
        ##tree = KDTree(target)
        distances, indices = tree.query(source)
        #print(indices)
        print( [(i,d) for d,i in zip(distances, indices)] )  
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

    apply_transformation = lambda points,R,t : (R @ points.T).T + t

    # Initialize variables
    src = source.copy()
    transformation1 = np.eye(4)

    mean_error = None 
    for i in range(max_iterations):
        #closest_points, distances = find_closest_points(src, target)
        closest_points, distances = find_closest_points_no_duplicates_iterative(src, target)
        
        R, t = compute_transformation(src, closest_points)
        src = apply_transformation(src, R, t)
        for j,(t2,s2,t1,d1) in enumerate(zip(target,source,src,distances)) : 
            diff = np.sqrt( np.sum((t2-t1)**2) )
            #print(i,j, s2,"->", t2, t1, d1, diff)

        # Update the global transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        transformation1 = T @ transformation1

        mean_error = np.mean(distances)
        print("its:", i, "mean_error:", mean_error)
        #print() 
        
        if mean_error < tolerance: break

    print("its:", i, "mean_error:", mean_error)
    

    matrix = mathutils.Matrix(transformation1.tolist())
    rotation_matrix = matrix.to_3x3()

    # Convert the rotation matrix to Euler angles (ZYX order by default)
    euler_angles = rotation_matrix.to_euler() 

    # Convert radians to degrees if needed
    euler_degrees = [np.rad2deg(angle) for angle in euler_angles]

    print("Euler angles in radians:", euler_angles)
    print("Euler angles in degrees:", euler_degrees)    
    return src, matrix

#------------------------------------------------------------------#
def get_transformation(obj1, obj2):
    # Ensure both objects are in world coordinates
    world_loc_1 = obj1.matrix_world.to_translation()
    world_loc_2 = obj2.matrix_world.to_translation()

    # Translation vector is the difference in location
    translation = world_loc_2 - world_loc_1
    print("[get_transformation] translation:\n", translation)
    
    # Extract rotation matrices
    rotation_matrix_1 = obj1.matrix_world.to_3x3()
    rotation_matrix_2 = obj2.matrix_world.to_3x3()

    # Compute the relative rotation matrix
    relative_rotation_matrix = rotation_matrix_1.inverted() @ rotation_matrix_2

    # Convert the relative rotation matrix back to Euler angles
    #rotation = relative_rotation_matrix.to_euler("XYZ")
    rotation_euler = relative_rotation_matrix.to_euler('XYZ') 
    euler = np.array([rotation_euler.x, rotation_euler.y, rotation_euler.z])
    print("[get_transformation] rotation_euler:\n", np.degrees(euler) )
    #print("rotation:", rotation_euler)

    #obj2_vertices = np.array([v.co for v in obj2.data.vertices])
    #obj1_vertices = np.array([v.co for v in obj1.data.vertices])
    #distances = np.linalg.norm(obj1_vertices - obj2_vertices, axis=1)
    #print("[get_transformation] distances:\n", np.sum(distances) )
    #return translation, rotation


#------------------------------------------------------------------#
def transform_and_apply(obj, euler_degrees, translation):
    bpy.ops.object.select_all(action='DESELECT')
    
    ## translate 
    translation_matrix = mathutils.Matrix.Translation(translation)
    print("[transform_and_apply] translation:\n", translation_matrix )

    ## rotate 
    rotation_euler_rads = np.radians( np.array(euler_degrees) ) 
    rotation_euler = mathutils.Euler(rotation_euler_rads, 'XYZ')
    rotation_matrix = rotation_euler.to_matrix().to_4x4()
    print("[transform_and_apply] rotation:\n", rotation_matrix )
    
    ## transform 
    transform_matrix = translation_matrix @ rotation_matrix
    print("[transform_and_apply] transform:\n", transform_matrix )
    
    obj.matrix_world = transform_matrix
    
    ## apply 
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    return obj


#------------------------------------------------------------------#
def copy_transform_and_apply(obj, rotation_euler_degrees, translation):
    bpy.ops.object.select_all(action='DESELECT')
    
    new_obj = obj.copy()
    new_obj.name = obj.name + "_copy"
    
    new_obj.data = obj.data.copy()
    new_obj.data.name = obj.data.name + "_copy"    
    bpy.context.collection.objects.link(new_obj)
    transform_and_apply(new_obj, rotation_euler_degrees, translation)
    
    """
    translation_matrix = mathutils.Matrix.Translation(translation)
    print("[transform_and_apply] translation:\n", translation_matrix )

    rotation_euler_degrees = np.radians( np.array(rotation_euler_degrees) ) 
    rotation_euler = mathutils.Euler(rotation_euler_degrees, 'XYZ')
    rotation_matrix = rotation_euler.to_matrix().to_4x4()
    print("[transform_and_apply] rotation:\n", rotation_matrix )
    
    transform_matrix = translation_matrix @ rotation_matrix
    print("[transform_and_apply] transform:\n", transform_matrix )
    
    new_obj.matrix_world = transform_matrix
    
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    """
    return new_obj



#------------------------------------------------------------------#
def compute_vertex_distances_and_apply2(rotation_euler, obj1, obj2):
    bpy.ops.object.select_all(action='DESELECT')
    
    rotation_euler = np.radians(np.array(rotation_euler))
    rotation_matrix = mathutils.Euler(rotation_euler, 'XYZ').to_matrix().to_4x4()
    
    obj2_matrix = obj2.matrix_world.copy()
    obj2.matrix_world = rotation_matrix @ obj2_matrix
    
    obj2.select_set(True)
    bpy.context.view_layer.objects.active = obj2
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    obj2_vertices = np.array([v.co for v in obj2.data.vertices])
    obj1_vertices = np.array([v.co for v in obj1.data.vertices])
    distances = np.linalg.norm(obj1_vertices - obj2_vertices, axis=1)
    #print( distances )
    
    return np.sum(distances)

#------------------------------------------------------------------#
def compute_vertex_distances_and_apply1(rotation_euler, obj1, obj2):
    bpy.ops.object.select_all(action='DESELECT')
    
    rotation_euler = np.radians(np.array(rotation_euler))
    rotation_matrix = mathutils.Euler(rotation_euler, 'XYZ').to_matrix().to_3x3()
    
    obj2_vertices = np.array([v.co for v in obj2.data.vertices])
    ## Apply inverse rotation!! 
    obj2_rotated = np.dot(obj2_vertices, rotation_matrix.transposed()) 
    
    obj1_vertices = np.array([v.co for v in obj1.data.vertices])
    distances = np.linalg.norm(obj1_vertices - obj2_rotated, axis=1)
    #print(distances)
    
    #for i, v in enumerate(obj2.data.vertices): v.co = obj2_rotated[i]    
    return np.sum(distances)


#------------------------------------------------------------------#
def align_objects(obj1, obj2):
    bpy.ops.object.select_all(action='DESELECT')
    
    ## Translate
    ##world_loc_1 = obj1.matrix_world.to_translation()
    ##world_loc_1 = obj1.matrix_world.translation
    world_loc_1 = obj1.location @ obj1.matrix_world
    print("[align_objects] world_loc_1:", world_loc_1) 
    
    ##world_loc_2 = obj2.matrix_world.to_translation()
    ##world_loc_2 = obj2.matrix_world.translation
    world_loc_2 = obj2.location @ obj2.matrix_world
    print("[align_objects] world_loc_2:", world_loc_2) 

    translation = world_loc_1 - world_loc_2
    obj2.location += translation 
    #return 
    
    #obj2.select_set(True)
    #bpy.context.view_layer.objects.active = obj2
    #bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    #return 

    ## Rotate 
    initial_rotation = np.zeros(3)  # No rotation initially
    
    options = {
        'maxiter': 100,  # Maximum number of iterations
        'ftol': 1e-6,    # Function value tolerance
        'xtol': 1e-6,    # Tolerance for parameters
        'disp': True      # Display optimization progress
    }    
    
    bounds = [(-90, 90), (-90, 90), (-90, 90)]
    #bounds = [(0, 180), (0, 180), (0, 180)]
    result = scipy.optimize.minimize(compute_vertex_distances_and_apply1, 
                                     initial_rotation, 
                                     args = (obj1, obj2), 
                                     bounds = bounds, 
                                     method = 'Powell', 
                                     options = options 
                                    )    

    #if result.fun < 1e-3 :
    #material = bpy.data.materials.new(name="Good")
    #material.diffuse_color = (0, 0, 1, 1.0)
    #bpy.context.object.active_material = None
    #obj2.data.materials.append(material)

    best_rotation_euler_deg = result.x
    best_rotation_euler_rad = np.deg2rad(result.x) 
    print(f"[align_objects] Optimal rotation (deg): {best_rotation_euler_deg} ({result.fun})") 
    
    rotation_euler = mathutils.Euler(best_rotation_euler_rad, 'XYZ')    
    print("[align_objects] align_objects:", rotation_euler)
    obj2.rotation_euler = rotation_euler
    
    obj2.select_set(True)
    bpy.context.view_layer.objects.active = obj2
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
     
    ## Display highline    
    ##bpy.context.object.show_wire = True
      
    bpy.context.object.display_type = 'WIRE'
    bpy.context.object.show_name = True
    return best_rotation_euler_deg


#------------------------------------------------------------------#
#------------------------------------------------------------------#
## X.0. 
clean_all_objects()
bpy.context.scene.cursor.location = (0, 0, 0)


def suzanne_test_a() : 
    bpy.ops.object.select_all(action='DESELECT')
    
    #bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), rotation=(0,0,0))
    bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0), rotation=(0,0,0))
    meshi = bpy.context.active_object
    meshi.name = "MySuzanne1"
    meshi.data.name = "MySuzanne1"
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    meshi.location = bpy.context.scene.cursor.location

    rotation = [0.0, 0.0, 0.0]
    rotation = [-30.0, 45.0, 0.0] # 258.8234485348652
    rotation = np.random.uniform(-90,90, size=3)

    translation = [0.0, 0.0, 1.0]
    translation = np.random.uniform(-1,1, size=3)
    meshj = copy_transform_and_apply(meshi, rotation, translation) 
    print("rotation:", rotation)
    print("translation:", translation)
        
    align_objects(meshj, meshi)
    return 


def lego_test_a() :
    bpy.ops.object.select_all(action='DESELECT')
     
    #filepath=r"F:\Models\USD\plum_blossom_10369.usd"
    filepath=r"F:\Models\USD\10369b1.usd"
        
    objects_lego = import_and_get_new_objects("usd_import", None, filepath=filepath)
    bpy.ops.object.select_all(action='DESELECT')

    meshes_lego = {obj.name:obj for obj in objects_lego if obj.type == "MESH"}
    print("meshes_lego:", len(meshes_lego) ) # 569 

    [object_origin_set(meshj) for meshj_name,meshj in meshes_lego.items()]

    meshi_name = "Base_322" # 3542 -> 1771 
    meshi = meshes_lego.get(meshi_name) 
        
    meshi.select_set(True) 
    bpy.context.view_layer.objects.active = meshi

    #bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.select_all(action='SELECT')  
    #bpy.ops.mesh.tris_convert_to_quads()
    #bpy.ops.object.mode_set(mode='OBJECT')
    #
    #decimate_mod = meshi.modifiers.new(name="Decimate", type='DECIMATE')
    #decimate_mod.ratio = 0.5 # 1771     
    #bpy.ops.object.modifier_apply(modifier=decimate_mod.name) 
    
    rotation = np.random.uniform(-90,90, size=3)
    translation = np.random.uniform(-1,1, size=3) # [0.0, 0.0, 1.0]
    meshj = copy_transform_and_apply(meshi, rotation, translation) 
    ##meshj.location = bpy.context.scene.cursor.location

    best_rotation_euler = align_objects(meshj, meshi) 
    return 

suzanne_test_a() 
lego_test_a() 


"""
meshi.select_set(True) 
bpy.context.view_layer.objects.active = meshi

bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0)})
meshj = bpy.context.active_object
meshj.name = meshi_name + "_copy"
meshj.data.name = meshj.name 
meshj.location = bpy.context.scene.cursor.location
#"""

"""
verticei = object_vertices_get(meshi)
verticej = object_vertices_get(meshj)
aligned_source, final_transformation = icp1(verticei, verticej)
print( final_transformation )

meshj.matrix_world = final_transformation #mathutils.Matrix(final_transformation.tolist())
meshj.select_set(True) 
"""
"""
#rotation = (30.0, 20.0, 10.0)
interations = 100
rotation_deg = np.random.uniform(-90,90, size=3)
rotation_deg = [-33.10178772, 12.97873411, 7.53109624] # :) 
rotation_deg = [75.5216435, 25.90979137, 72.87189898] # :( 
#
rotation = mathutils.Euler(np.deg2rad(rotation_deg), 'XYZ') 
#bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0), rotation=rotation)
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0), rotation=rotation)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

meshj = bpy.context.active_object
meshj.name = "MySuzanne2"
meshj.data.name = "MySuzanne2"


verticei = object_vertices_get(meshi)
verticej = object_vertices_get(meshj)
aligned_source, final_transformation = icp1(verticei, verticej, interations)
print("final_transformation:", final_transformation )
print("rotation:", rotation_deg)

##meshj.matrix_world = mathutils.Matrix(final_transformation.tolist()) 
##final_transformation *= -1
meshj.rotation_euler = final_transformation.to_euler()
#meshj.select_set(True) 
"""



"""
bpy.context.scene.cursor.location = (0, 0, 0)


meshi = bpy.data.objects.get("object")
meshi.select_set(True) 
bpy.context.view_layer.objects.active = meshi
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
#bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
meshi.location = bpy.context.scene.cursor.location

meshj = bpy.data.objects.get("reference")
meshj.select_set(True) 
bpy.context.view_layer.objects.active = meshj
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
#py.ops.view3d.snap_selected_to_cursor(use_offset=False)
meshj.location = bpy.context.scene.cursor.location



verticei = object_vertices_get(meshi)
verticej = object_vertices_get(meshj)
aligned_source, final_transformation = icp(verticej, verticei)
print( final_transformation )

meshj.matrix_world = mathutils.Matrix(final_transformation.tolist())
meshj.select_set(True) 
"""

## X.0. 
"""
for j,(meshj_name,meshj) in enumerate(meshes_lego.items()) : 
    #print(j, meshi, meshj)
    
    object_origin_set(meshj)    
    verticej = object_vertices_get(meshj)
    
    if( verticei.shape == verticej.shape ): 
        print(j, meshi, meshj)
        aligned_source, final_transformation = icp(verticei, verticej)
        print("Final Transformation Matrix:")
        print(final_transformation)        

        #meshj.matrix_world = mathutils.Matrix(final_transformation.tolist())
        #meshj.select_set(True) 
"""

#aligned_source, final_transformation = icp(source, target)
#print("Final Transformation Matrix:")
#print(final_transformation)







print("ok!")
#------------------------------------------------------------------#
#------------------------------------------------------------------#