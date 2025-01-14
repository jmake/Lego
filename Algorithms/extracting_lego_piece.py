import bpy 

#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
def clean_all_objects():
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False)
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.context.scene.cursor.location = (0, 0, 0)
    return 


def object_origin_set(obj) : 
    bpy.ops.object.select_all(action='DESELECT')

    obj.select_set(True) 
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    ##bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN') # BOUNDS

    obj.select_set(False) 
    bpy.context.view_layer.objects.active = None
    return 


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


def gltf_load(gltf_file_path) :
    clean_all_objects()
    #gltf_file_path = r"F:\Models\porsche_911_rsr_42096.glb"
    bpy.ops.import_scene.gltf(filepath=gltf_file_path)
    
    bpy.ops.object.select_all(action='DESELECT')

    objects_lego = bpy.data.objects
    meshes_lego = {obj.name:obj for obj in objects_lego if obj.type == "MESH"}
    print("meshes_lego:", len(meshes_lego) )  

    [object_origin_set(meshj) for meshj_name,meshj in meshes_lego.items()]

    return meshes_lego


## X.0. 
#--------------------------------------------------------------------------||--#

gltf_file_path = r"F:\Models\porsche_911_rsr_42096.glb"
meshes = gltf_load(gltf_file_path) 

objects_lego = bpy.data.objects
meshes_lego = {obj.name:obj for obj in objects_lego if obj.type == "MESH"}
print("meshes_lego:", len(meshes_lego) )  




print("ok!")
#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
