import os
import bpy
import math

import mathutils
from mathutils import Vector

#import open3d as o3d

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
#------------------------------------------------------------------#
filepath=r"F:\Models\USD\plum_blossom_10369.usd"
objects_lego = import_and_get_new_objects("usd_import", None, filepath=filepath)

meshes_lego = {obj.name:obj for obj in objects_lego if obj.type == "MESH"}
print("meshes_lego:", len(meshes_lego) ) # 569 


mesh_name = "Base.195" 

mesh = meshes_lego.get(mesh_name) 
print(mesh)
save_vertices_as_ply(mesh_name, "reference.ply")

object_origin_set(mesh_name)
object_rotate(mesh_name, 45, orient_axis='Z')
object_rotate(mesh_name, 30, orient_axis='X')

save_object_as_glb(mesh_name, "object.glb")
save_vertices_as_ply(mesh_name, "object.ply")


print("ok!")
#------------------------------------------------------------------#
#------------------------------------------------------------------#
r"""
import sys
print(sys.executable)

python.exe --version

"""

r"""
python.exe -m venv Blender1

.\Blender1\Scripts\Activate.ps1

python.exe -m pip install open3d
python.exe -m pip install numpy 




"""

r"""

C:\Users\zvl_2\AppData\Local\ov\pkg\blender-4.2.0-usd.202.1\omni.blender.launcher.bat

"""

#------------------------------------------------------------------#
#------------------------------------------------------------------#