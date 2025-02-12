import bpy 
import numpy as np 


import bmesh
from collections import defaultdict


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
def frame_rate_get() : 
    fps = bpy.context.scene.render.fps
    fps_base = bpy.context.scene.render.fps_base  
    frame_rate = fps / fps_base 
    return frame_rate 


def scene_frames(frame_rate, n_frames, play) : 
    start_frame = int(frame_rate * 1) 
    
    bpy.context.scene.frame_end = n_frames
    bpy.context.scene.frame_set(start_frame) 
    bpy.context.scene.frame_current = start_frame
    
    current_frame = bpy.context.scene.frame_current
    total_time = n_frames / frame_rate / 60.0 
    
    msg  = f"{frame_rate} {n_objects} "
    msg += f"{n_frames} {total_time}"     
    print(msg)

    if play : bpy.ops.screen.animation_play()
    return 


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
def objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    objects = np.array(collection.objects)
    
    names = [obj.name for obj in objects]
    ordered = np.argsort(names) 
    results = objects[ordered]
    return results


def get_objects_by_material(objects):
    selected = defaultdict()

    for i, obj in enumerate(objects) :
        msg  = f" '{obj.name}', '{obj.type}'" 
        
        parent = obj.parent 
        if parent : 
            msg += f" '{parent.name}', '{parent.type}'"

        children = obj.children  
        msg += " children:"
        if children : 
            msg += " ".join([c.name for c in children])

        #print(i, msg)

        if children : 
            types = np.array([c.type for c in children]) 
            names = np.array([c.name for c in children]) 
            if np.all(types == 'MESH') : 
                if len(names) > 1 :
                    print( names )
                    selected[obj] = children

    return dict(selected)


#------------------------------------------------------------------------#
def merge_meshes(objects, merged_name) :
    data = bpy.data.meshes.get(merged_name) 
    if data : 
        bpy.data.meshes.remove(data, do_unlink=True) 

    obj = bpy.data.objects.get(merged_name) 
    if obj : 
        bpy.data.objects.remove(obj, do_unlink=True) 

    mesh_data = bpy.data.meshes.new(merged_name)
    merged_obj = bpy.data.objects.new(merged_name, mesh_data)
    bpy.context.collection.objects.link(merged_obj)

    bm = bmesh.new()
    for obj in objects:
        if obj.type == 'MESH':
            temp_mesh = obj.to_mesh()
            temp_mesh.transform(obj.matrix_world)
            bm.from_mesh(temp_mesh)
            #bpy.data.meshes.remove(temp_mesh)

    bm.to_mesh(mesh_data)
    bm.free()
    return merged_obj


def merge_and_parent_meshes(objects, apply) :
    selected = get_objects_by_material( objects ) 
    for obj,children in selected.items() : 
        obj.select_set(True) 
        #bpy.context.view_layer.objects.active = obj

        merged = merge_meshes(children, "merged_" + obj.name) 
        
        if apply : 
            merged.parent = obj 
            merged.matrix_basis = merged.matrix_world @ obj.matrix_world.inverted()

            merged.select_set(True) 
            bpy.context.view_layer.objects.active = merged
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            #merged.select_set(False) 
            print( merged )

        for c in children : 
            print(obj.name,c)
            #c.select_set(True) 
            c.parent = None 

    bpy.context.view_layer.objects.active = None 
    return 

#------------------------------------------------------------------------#



#------------------------------------------------------------------------#



#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
## X.1. 
bpy.ops.outliner.orphans_purge(do_recursive=True)


collection_name = "Model"
objects = objects_in_collection( collection_name )
n_objects = len(objects)

merge_and_parent_meshes(objects, "True") 



## X.1. 
frame_rate = frame_rate_get() 
n_frames = int(n_objects * frame_rate) 

scene_frames(frame_rate, n_frames, False) 


print("ok!")
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
