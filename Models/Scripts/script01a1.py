import bpy 
import numpy as np 

target_name = "CameraD2_Target" 
target_name = "CameraD2_DoF"
target_obj = bpy.data.objects.get(target_name)


collection_name = "Model"



def list_objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    objects = np.array(collection.objects)
    
    names = [obj.name for obj in objects]
    ordered = np.argsort(names) 
    results = objects[ordered]
    return results



def print_object_position(scene):
    obj = bpy.data.objects.get(obj_name)
    print(f"Frame {scene.frame_current}: Object Position: {obj.location}")



def print_current_frame_time(scene):
    current_frame = scene.frame_current
    fps = scene.render.fps
    current_time = current_frame / fps 
    #print(f"Frame {current_frame}: Time {current_time} seconds")

    itime = np.floor(current_time)
    itime = np.clip(itime, 0, len(objects)-1)
    itime = int(itime)

    obj = objects[itime] 
    position = obj.location @ obj.matrix_world
    target_obj.location = position 
    print(current_frame, current_time, itime, obj.name, position )        
    return 



def load_mp3(frames):
    filepath = r"F:\IKEA\z2024_2\10369\Models\Tools\lego-piece-pressed-105360.mp3" 

    original_area = bpy.context.area.type     
    bpy.context.area.type = "SEQUENCE_EDITOR"

    sequence_editor = bpy.context.scene.sequence_editor
    for strip in sequence_editor.sequences_all :
        sequence_editor.sequences.remove(strip) 

    for frame_start in frames : 
        bpy.ops.sequencer.sound_strip_add(filepath=filepath, frame_start=frame_start, channel=1)
    
    # Restore the original area type
    bpy.context.area.type = original_area
    return 


def create_nurbs_path(name, points, resolution_u):
    curve_obj = bpy.data.objects.get(name) 
    if curve_obj : 
        bpy.data.objects.remove(curve_obj, do_unlink=True) 
    
    curve_data = bpy.data.curves.new(type='CURVE', name=name)
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 50 


    spline = curve_data.splines.new(type='NURBS')
    spline.points.add(count=len(points)-1)

    for i, point in enumerate(points):
        spline.points[i].co = (point[0], point[1], point[2], 1)


    curve_data.use_path = True 
    curve_data.path_duration =  1
    curve_data.use_path_clamp = True 

    spline.order_u = 32 
    spline.use_bezier_u = True
    spline.resolution_u = 32
    spline.use_endpoint_u = True

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    
    curve_obj.select_set(True) 
    bpy.context.view_layer.objects.active = curve_obj 
    return curve_obj



objects = list_objects_in_collection(collection_name)
n_objects = len(objects)


fps = bpy.context.scene.render.fps
fps_base = bpy.context.scene.render.fps_base  
frame_rate = fps / fps_base 
n_frames = int(n_objects * frame_rate) 

points = [o.location for o in objects]

path_obj = create_nurbs_path("PathDX", points, resolution_u=1)
path_obj.data.eval_time = 0.0
path_obj.data.keyframe_insert(data_path="eval_time", frame=0)

path_obj.data.eval_time = 1.0
path_obj.data.keyframe_insert(data_path="eval_time", frame=n_frames)

target_obj = bpy.data.objects["CameraD3_Target"] 
target_obj.constraints["Follow Path"].target = path_obj
target_obj.constraints["Follow Path"].use_curve_follow = True 



handlers = bpy.app.handlers.frame_change_post 
handlers.clear()

#handlers.append(print_object_position)
handlers.append(print_current_frame_time)
print(f"Handlers:{len(handlers)}" )


current_frame = bpy.context.scene.frame_current
#load_mp3( [i*fps for i in range(len(objects))] )

#total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1 
total_time = n_frames / fps / 60.0 
print(fps, n_objects, n_frames, total_time )



if 1 : 
    start_frame = int(frame_rate * 1) 
    bpy.context.scene.frame_end = n_frames
    bpy.context.scene.frame_set(start_frame) 
    bpy.context.scene.frame_current = start_frame
    #bpy.ops.screen.animation_play()

print("ok!")