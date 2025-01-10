import bpy
import mathutils

from mathutils import Vector


def are_bounding_boxes_overlapping(bbox1, bbox2):
    #print(bbox1, bbox2) 
    min1 = Vector(map(min, *bbox1))
    max1 = Vector(map(max, *bbox1))
    min2 = Vector(map(min, *bbox2))
    max2 = Vector(map(max, *bbox2))

    return (
        min1.x <= max2.x and max1.x >= min2.x and
        min1.y <= max2.y and max1.y >= min2.y and
        min1.z <= max2.z and max1.z >= min2.z
    )


def get_bounding_box_world_coords(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def object_remove(name): 
    existing_obj = bpy.data.objects.get(name)
    if existing_obj : 
        bpy.data.objects.remove(existing_obj, do_unlink=True)


def create_bounding_box(target, parent):
    if target is None :
        raise ValueError("No object provided.")

    mesh_name = f"{target.name}_BoundingBoxMesh"
    mesh = bpy.data.meshes.get(mesh_name)
    if mesh: bpy.data.meshes.remove(mesh) 
    bbox_mesh = bpy.data.meshes.new(mesh_name)
        
    obj_name = f"{target.name}_BoundingBox"
    obj = bpy.data.objects.get(obj_name)
    if obj : bpy.data.objects.remove(obj, do_unlink=True)    
    bbox_obj = bpy.data.objects.new(obj_name, bbox_mesh)
    
    bpy.context.collection.objects.link(bbox_obj)    
    bbox_corners = [target.matrix_world @ mathutils.Vector(corner) for corner in target.bound_box]
    verts = bbox_corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    bbox_mesh.from_pydata(verts, edges, [])
    bbox_mesh.update()
    
    if parent : 
        bbox_obj.parent = parent 
        
    return bbox_obj


def delete_objects_in_hierarchy(parent_name):
    parent_obj = bpy.data.objects.get(parent_name)
    
    if parent_obj:
        def delete_recursive(obj):
            children = [child for child in bpy.data.objects if child.parent == obj]

            for child in children: delete_recursive(child)
            bpy.data.objects.remove(obj, do_unlink=True)
        
        delete_recursive(parent_obj)
    else:
        print(f"Parent object '{parent_name}' not found.")



#------------------------------------------------------------------#
def create_or_replace_empty(name, collection):
    existing_obj = bpy.data.objects.get(name)
    if existing_obj and existing_obj.type == 'EMPTY':
        bpy.data.objects.remove(existing_obj, do_unlink=True)
    
    empty = bpy.data.objects.new(name=name, object_data=None)
    #bpy.context.scene.collection.objects.link(empty)
    
    for col in empty.users_collection: col.objects.unlink(empty)
    collection.objects.link(empty)
    #bbox_obj.select_set(True) 
    return empty


def create_collection(collection_name):
    
    # Check if collection exists
    collection = bpy.data.collections.get(collection_name)
    
    if collection:
        # Remove the collection and all its objects
        for obj in collection.objects:
            collection.objects.unlink(obj)
        bpy.data.collections.remove(collection)
        print(f"Existing collection '{collection_name}' deleted.")

    # Create a new collection
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    
    print(f"Collection '{collection_name}' created and objects linked.")
    return collection

#------------------------------------------------------------------#
#------------------------------------------------------------------#
r"""
object_pointers = {obj.as_pointer(): obj for obj in bpy.data.objects}
meshes = {k:v for k,v in object_pointers.items() if v.type == "MESH"}
bboxes = {k:get_bounding_box_world_coords(v) 
            for k,v in meshes.items()} 
"""
nodes = [] 

def children_get(root_obj) :     
    parent_id = None 
    parent_name = None  
    parent = root_obj.parent    
    if parent : 
        parent_name = parent.name 
        parent_id = parent.as_pointer()
    
    children = root_obj.children
    node = (parent_name, root_obj.name)
    node = (parent_id, root_obj.as_pointer() )
    global nodes 
    nodes.append(node) 
    
    if children : 
        return [children_get(c) for c in children] 
    else : 
        return root_obj



def children_recursive_get(root_obj) :  

    children_get(root_obj)   
    for i,node in enumerate(nodes) :
        parent,child = node
        child = object_pointers.get(child)
        parent = object_pointers.get(parent)
        #print(i, parent, child.name)
        if child.type == "EMPTY" : 
            pass 
        else :
            print(i, child.type )
            #create_bounding_box(child) 
    return 


def select_mesh_objects_in_hierarchy(root_name):
    bpy.ops.object.select_all(action='DESELECT')

    objects = bpy.data.objects 
    root_obj = objects.get(root_name) 
   
    children_recursive_get(root_obj) 
    

def object_link(obj, collection) : 
    for col in obj.users_collection: col.objects.unlink(obj)
    collection.objects.link(obj)


def overlapping_get(root_name, meshes, collection) : 
    bpy.ops.object.select_all(action='DESELECT')
    
    empty = create_or_replace_empty(f"{root_name}_Empty", collection)
    
    objects = bpy.data.objects 
    obj_i = objects.get(root_name)
    obj_i.select_set(True) 

    bbox_obj = create_bounding_box(obj_i, empty) 
    #for col in bbox_obj.users_collection: col.objects.unlink(bbox_obj)
    #collection.objects.link(bbox_obj)
    bbox_obj.select_set(True) 
    object_link(bbox_obj, collection)

    bboxes = {k:get_bounding_box_world_coords(v) 
            for k,v in meshes.items()} 

    bbox_i = bboxes.get(root_name) 
    for i,(id_j,bbox_j) in enumerate(bboxes.items()) : 
        overlap = are_bounding_boxes_overlapping(bbox_i, bbox_j)
        if overlap : 
            obj_j = meshes.get(id_j) 
            #print(i, obj_i, obj_j)
            
            bbox_obj = create_bounding_box(obj_j, empty) 
            #for col in bbox_obj.users_collection: col.objects.unlink(bbox_obj)
            #collection.objects.link(bbox_obj)
            bbox_obj.select_set(True) 
            object_link(bbox_obj, collection)
                    

def import_and_get_new_objects(import_operator, collection, **kwargs):
    # Store current objects
    existing_objects = set(bpy.data.objects)
    
    # Perform the import
    getattr(bpy.ops.wm, import_operator)(**kwargs)
    
    # Get new objects
    new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
    
    for obj in new_objects : 
        for col in obj.users_collection: col.objects.unlink(obj)
        collection.objects.link(obj)
        
    return new_objects
    

def clean_all_objects():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


#------------------------------------------------------------------#
#------------------------------------------------------------------#
## 0.0.
bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False)
clean_all_objects()


collection1 = create_collection("Lego") 
collection2 = create_collection("BBoxes") 


#r"""
#object_name = "World" 
#delete_objects_in_hierarchy(object_name)

## 1.0. 
filepath=r"F:\Models\USD\plum_blossom_10369.usd"
objects_lego = import_and_get_new_objects("usd_import", collection1, filepath=filepath)
bpy.ops.object.select_all(action='DESELECT')
print( len(objects_lego) ) # 1536
#"""0

meshes_lego = {obj.name:obj for obj in objects_lego if obj.type == "MESH"}
print( len(meshes_lego) ) # 569 

## 2.0. 
object_name = "Base.195" 
overlapping_get(object_name, meshes_lego, collection2)  


object_name = "Base.312" 
overlapping_get(object_name, meshes_lego, collection2)  



## X.0. 





## X.0. 
objects_all = bpy.data.objects 

