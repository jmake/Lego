import vtk
import numpy as np
from sklearn.decomposition import PCA

import vtk

import vtk

def visualize_ply_file(ply_file):
    """Visualize the PLY file using VTK."""
    # Load the PLY file
    reader = vtk.vtkPLYReader()
    reader.SetFileName(ply_file)
    reader.Update()

    # Check if the file was successfully loaded
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        print("Error: The PLY file contains no points or failed to load.")
        return

    # Create a mapper for the polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # Create an actor to represent the mesh
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Set background color

    # Set camera to focus on the loaded object
    renderer.ResetCamera()

    # Start the interaction
    render_window.Render()
    render_window_interactor.Start()



def load_ply_file(ply_file):
    """Load the PLY file and return the vtkPolyData."""
    reader = vtk.vtkPLYReader()
    reader.SetFileName(ply_file)
    reader.Update()
    return reader.GetOutput()

def extract_points(polydata):
    """Extract points from vtkPolyData and return as a numpy array."""
    points = polydata.GetPoints()
    return np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

def perform_pca(points):
    """Perform PCA on the points and return the axes."""
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_

def compute_obb_axes(ply_file):
    """Compute the OBB axes (X, Y, Z) based on PCA of points in the PLY file."""
    # Load and read the PLY file
    polydata = load_ply_file(ply_file)

    # Extract points from the polydata
    np_points = extract_points(polydata)

    # Perform PCA on the points
    axes = perform_pca(np_points)

    return axes
    # Return the axes (X, Y, Z)
    #return axes[0], axes[1], axes[2]


ply_file = r"F:\Models\Blender\Example\base_322b.ply"

(x_axis, y_axis, z_axis) = compute_obb_axes(ply_file)
print("X-axis:", x_axis)
print("Y-axis:", y_axis)
print("Z-axis:", z_axis)


reader = vtk.vtkPLYReader()
reader.SetFileName(ply_file)
reader.Update()
polydata = reader.GetOutput()

# Mapper and actor for the STL geometry
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Renderer and window setup
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0, 0, 0)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

renderer.ResetCamera()

# Compute OBB
obb_tree = vtk.vtkOBBTree()
corner = [0.0, 0.0, 0.0]
max_vec = [0.0, 0.0, 0.0]
mid_vec = [0.0, 0.0, 0.0]
min_vec = [0.0, 0.0, 0.0]
size = [0.0, 0.0, 0.0]
obb_tree.ComputeOBB(reader.GetOutput().GetPoints(), corner, max_vec, mid_vec, min_vec, size)


print("vtkOBBTree:")
print(f"corner: {corner}")
print(f"max: {max_vec}")
print(f"mid: {mid_vec}")
print(f"min: {min_vec}")
print(f"size: {size}")

# Draw OBB axes
real_size = [np.linalg.norm(max_vec), np.linalg.norm(mid_vec), np.linalg.norm(min_vec)]
max_vec = np.array(max_vec) / real_size[0]
mid_vec = np.array(mid_vec) / real_size[1]
min_vec = np.array(min_vec) / real_size[2]

obb_axes = np.array([max_vec, mid_vec, min_vec])
aabb_axes = np.eye(3) 
rotation_matrix = np.dot(np.linalg.inv(obb_axes), aabb_axes)
print(rotation_matrix)


def add_line(renderer, start, end, color):
    line = vtk.vtkLineSource()
    line.SetPoint1(start)
    line.SetPoint2(end)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(1)
    renderer.AddActor(actor)
    return line 

corner_np = np.array(corner)
line1 = add_line(renderer, corner, corner_np + max_vec * real_size[0], [1, 0, 0])
line2 = add_line(renderer, corner, corner_np + mid_vec * real_size[1], [0, 1, 0])
line3 = add_line(renderer, corner, corner_np + min_vec * real_size[2], [0, 0, 1])


from vtk.util import numpy_support
points = polydata.GetPoints()
points = numpy_support.vtk_to_numpy(points.GetData())

apply_rotation = lambda points, rotation_matrix : np.dot(rotation_matrix, points.T).T
aligned_points = apply_rotation(points, rotation_matrix)

vtk_points = vtk.vtkPoints()
for point in aligned_points: vtk_points.InsertNextPoint(point)

vtk_poly_data = vtk.vtkPolyData()
vtk_poly_data.SetPoints(vtk_points)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(vtk_poly_data)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
renderer.AddActor(actor)

ply_writer = vtk.vtkPLYWriter()
ply_writer.SetFileName("output.ply")
ply_writer.SetInputData(vtk_poly_data)  # Set the vtkPolyData as input
ply_writer.Write()

#exit()
# Render and start interaction
render_window.Render()
interactor.Start()