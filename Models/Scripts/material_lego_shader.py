import bpy 
import numpy as np 

from collections import defaultdict


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
def objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    objects = np.array(collection.objects)
    
    names = [obj.name for obj in objects]
    ordered = np.argsort(names) 
    results = objects[ordered]
    return results


def meshes_in_collection(collection_name): 
    objects = objects_in_collection(collection_name)
    objects = np.array([obj for obj in objects if obj.type == 'MESH'])
    return objects


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
def get_objects_by_material(objects):
    material_dict = defaultdict(list)

    for obj in objects:
        if obj.type == 'MESH' and obj.data.materials:
            for mat in obj.data.materials:
                if mat:
                    material_dict[mat.name].append(obj.name)

    return dict(material_dict)


#------------------------------------------------------------------------#
def get_base_color(obj):
    #obj = bpy.data.objects.get(obj_name)
    if not obj or not obj.data.materials:
        return None

    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    return node.inputs['Base Color'].default_value[:3]

    return None


def setup_base_color_with_rgb_node(obj):
    #obj = bpy.data.objects.get(obj_name)
    if not obj or not obj.data.materials:
        return None

    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            principled_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not principled_node:
                continue

            rgb_node = next((n for n in nodes if n.type == 'RGB'), None)
            if not rgb_node:
                rgb_node = nodes.new(type='ShaderNodeRGB')
                rgb_node.location = (-300, 0)  

            # Copy the Base Color from the Principled BSDF to the RGB node
            base_color = principled_node.inputs['Base Color'].default_value[:3]
            rgb_node.outputs['Color'].default_value = base_color + (1.0,)  # Add alpha channel (1.0 for opaque)

            # Disconnect existing links and connect the RGB node
            base_color_input = principled_node.inputs['Base Color']
            for link in links:
                if link.to_socket == base_color_input:
                    links.remove(link)

            links.new(rgb_node.outputs['Color'], base_color_input)

            return rgb_node.outputs['Color'].default_value[:3]

    return None


def setup_base_color_with_rgb_node(obj):
    if not obj or not obj.data.materials:
        return None

    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            principled_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if not principled_node:
                continue

            rgb_node = next((n for n in nodes if n.type == 'RGB'), None)
            if not rgb_node:
                rgb_node = nodes.new(type='ShaderNodeRGB')
                rgb_node.location = (-300, 0)

            base_color = principled_node.inputs['Base Color'].default_value[:3]
            rgb_node.outputs['Color'].default_value = base_color + (1.0,)

            mix_shader = next((n for n in nodes if n.type == 'ShaderNodeMixShader'), None)
            if not mix_shader:
                mix_shader = nodes.new(type='ShaderNodeMixShader')
                mix_shader.location = (0, 200)

            material_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if not material_output:
                material_output = nodes.new(type='ShaderNodeOutputMaterial')
                material_output.location = (300, 0)

            base_color_input = principled_node.inputs['Base Color']
            for link in links:
                if link.to_socket == base_color_input:
                    links.remove(link)

            if mix_shader.inputs[1].is_linked:
                links.remove(mix_shader.inputs[1].links[0])

            links.new(rgb_node.outputs['Color'], base_color_input)
            links.new(principled_node.outputs['BSDF'], mix_shader.inputs[1])
            links.new(mix_shader.outputs[0], material_output.inputs['Surface'])

            return rgb_node.outputs['Color'].default_value[:3]

    return None

def get_unique_materials(objects):
    unique_materials = set()

    for obj in objects:
        if obj.data.materials:
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    unique_materials.add(mat)

    return list(unique_materials)




def setup_base_color_with_rgb_node_for_materials(objects):
    unique_materials = get_unique_materials(objects)

    for mat in unique_materials:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        principled_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not principled_node:
            continue

        rgb_node = next((n for n in nodes if n.type == 'RGB'), None)
        if not rgb_node:
            rgb_node = nodes.new(type='ShaderNodeRGB')
            rgb_node.location = (-300, 0)

        base_color = principled_node.inputs['Base Color'].default_value[:3]
        rgb_node.outputs['Color'].default_value = base_color + (1.0,)

        mix_shader = next((n for n in nodes if n.type == 'ShaderNodeMixShader'), None)
        if not mix_shader:
            mix_shader = nodes.new(type='ShaderNodeMixShader')
            mix_shader.location = (0, 200)
            mix_shader.inputs["Fac"].default_value = 0.75 

        shader_name = "LegoShader"
        lego_shader = next((n for n in nodes if n.name == shader_name), None)
        if not lego_shader:
            lego_shader = nodes.new(type='ShaderNodeGroup')
            lego_shader.node_tree = bpy.data.node_groups[shader_name]
            lego_shader.location = (-200, 0)

            lego_shader.inputs['Damage'].default_value = 1.0
            lego_shader.inputs['Random'].default_value = 0.0
            lego_shader.inputs['Translucent'].default_value = 0.0


        material_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not material_output:
            material_output = nodes.new(type='ShaderNodeOutputMaterial')
            material_output.location = (300, 0)

        base_color_input = principled_node.inputs['Base Color']
        for link in links:
            if link.to_socket == base_color_input:
                links.remove(link)

        if mix_shader.inputs[1].is_linked:
            links.remove(mix_shader.inputs[1].links[0])

        links.new(rgb_node.outputs['Color'], base_color_input)
        links.new(rgb_node.outputs['Color'], lego_shader.inputs['Color'])
        links.new(principled_node.outputs['BSDF'], mix_shader.inputs[1])
        links.new(lego_shader.outputs['Shader'], mix_shader.inputs[2])
        links.new(mix_shader.outputs[0], material_output.inputs['Surface'])

    return


#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
## X.1. 
collection_name = "Model"
objects = objects_in_collection(collection_name)
n_objects = len(objects)


meshes = meshes_in_collection(collection_name) 
n_meshes = len(meshes)

for mesh in meshes : 
    color = get_base_color( mesh )
    #setup_base_color_with_rgb_node( mesh )
    #print( mesh.name, color )

setup_base_color_with_rgb_node_for_materials(meshes) 


## X.1. 
materials_dict = get_objects_by_material(meshes)
print(materials_dict.keys())


## X.1. 



## X.1. 
frame_rate = frame_rate_get() 
n_frames = int(n_objects * frame_rate) 

scene_frames(frame_rate, n_frames, False) 



print("ok!")
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
