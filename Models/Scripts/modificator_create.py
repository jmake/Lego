import bpy 
import numpy as np 

import bmesh
from collections import defaultdict

bpy.ops.outliner.orphans_purge(do_recursive=True)


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
def objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    objects = np.array(collection.objects)
    
    names = [obj.name for obj in objects]
    ordered = np.argsort(names) 
    results = objects[ordered]
    return results


def collections_in_collection(collection_name) :
    collection = bpy.data.collections.get(collection_name)

    selected = defaultdict() 
    for c in collection.children : 
        children = objects_in_collection( c.name ) 
        selected[c] = children
    return selected 


#------------------------------------------------------------------------#
def vertice_create( name ):
    data = bpy.data.meshes.get(name)
    if data : 
        bpy.data.meshes.remove(data, do_unlink=True)

    obj = bpy.data.objects.get(name)
    if obj : 
        bpy.data.objects.remove(obj, do_unlink=True)

    mesh_data = bpy.data.meshes.new(name)

    bm = bmesh.new()
    bm.verts.new((0, 0, 0))
    bm.to_mesh(mesh_data)
    bm.free()

    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj


def modifier_create(name) : 
    vertice = vertice_create( name )

    mod_name = "GeometryNodes"
    mod = vertice.modifiers.new(name=mod_name, type='NODES')

    group_name = "GeoNode2"
    group = bpy.data.node_groups.get(group_name)
    mod.node_group = group
    return vertice 



#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
## X.1. 
collection_name = "Model"
collections = collections_in_collection(collection_name) 
n_collections = len(collections)
print(f"n_collections:{n_collections}")


## X.1. 
n_objects = 0 
for i,(collection,objects) in enumerate(collections.items()) : 
    print(f"{i:02}) [{collection.name}] len:{len(objects)} ({n_objects})")

    obj = modifier_create(f"vertice{i:02}")
    node = obj.modifiers["GeometryNodes"]
    node["Socket_3"] = collection
    node["Socket_4"] = n_objects
    n_objects += len(objects) 

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.geometry_nodes_input_attribute_toggle(input_name="Socket_4", modifier_name="GeometryNodes")

## force update!!
bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
bpy.context.view_layer.update()

print(f"n_objects:{n_objects}")



## X.1. 
frame_rate = 1 #frame_rate_get() 
n_frames = int(n_objects * frame_rate) 
scene_frames(frame_rate, n_frames, False) 


print("ok!")
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#