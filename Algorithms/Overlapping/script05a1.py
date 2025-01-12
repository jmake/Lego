import bpy 
import vtk 
import numpy as np
from vtk.util import numpy_support

from vtkmodules.vtkCommonColor import vtkNamedColors
colors = vtk.vtkNamedColors()

## numpy_support.vtk_to_numpy(_pd.GetArray(k))
## pymesh_object = bpy_to_pymesh_in_memory("Cube")
vtk.vtkCollisionDetectionFilter()

color_names = np.array( colors.GetColorNames().split('\n') )
#print( color_names )
#print( len(nc.GetColorNames() ))
#exit()

#--------------------------------------------------------------------------||--#
def bpy_to_pymesh_in_memory(obj_name):
    # Get the Blender object
    obj = bpy.data.objects[obj_name]
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found in Blender.")
    
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj_name}' is not a mesh.")
    
    # Ensure the object's data is updated
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Extract vertices and faces
    vertices = np.array([vert.co[:] for vert in obj.data.vertices])
    faces = np.array([face.vertices[:] for face in obj.data.polygons])
    
    # Create a PyMesh object
    pymesh_mesh = pymesh.form_mesh(vertices, faces)
    
    return pymesh_mesh


#--------------------------------------------------------------------------||--#
class Plotter : 
    def __init__(self) : 

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(colors.GetColor3d("SlateGray"))
        return 


    def ActorAdd(self, obj, color, opacity, text=None) : 
        cubeMapper = vtk.vtkPolyDataMapper()
        ##cubeMapper.SetInputConnection( obj.GetOutputPort() )

        try : 
            cubeMapper.SetInputData(obj)
        except : 
            cubeMapper.SetInputConnection( obj.GetOutputPort() )

        cubeActor = vtk.vtkActor()
        cubeActor.SetMapper(cubeMapper)
        cubeActor.GetProperty().SetOpacity(opacity) 
        cubeActor.GetProperty().SetColor(color)

        self.renderer.AddActor(cubeActor)
        #if text : self.TestAdd(obj, text, color)
        return 


    def VerticesAdd(self, verticesPolydata, color) :  
        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputData( verticesPolydata )
        vertexGlyphFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color) 
        actor.GetProperty().SetPointSize(5) 
        self.renderer.AddActor( actor )
        return 


    def TestAdd(self, obj, text, color) : 
        mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(cube.GetOutputPort())
        mapper.SetInputData(obj)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a text actor for the mesh name
        mesh_name = text
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(mesh_name)
        text_actor.GetTextProperty().SetColor(1, 1, 1)  # White text
        text_actor.GetTextProperty().SetFontSize(20)
        text_actor.GetProperty().SetColor(color)

        # Position the text relative to the cube
        bounds = actor.GetBounds()
        text_actor.SetPosition(bounds[0], bounds[3])  # X=minX, Y=maxY
        self.renderer.AddActor( text_actor )
        return 



    def Show(self): 
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(self.renderer)
        renderWindow.SetWindowName("WindowName")

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderWindow.Render()
        renderWindowInteractor.Start()
        return 

#--------------------------------------------------------------------------||--#
def create_suzanne(name):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.mesh.primitive_monkey_add()
    suzanne = bpy.context.active_object
    suzanne.name = name
    return suzanne

#--------------------------------------------------------------------------||--#
def write_vtk_to_ply(vtk_polydata, filename):
    ply_writer = vtk.vtkPLYWriter()
    ply_writer.SetInputData(vtk_polydata)
    ply_writer.SetFileName(filename)
    ply_writer.Write()
    print(f"File saved as {filename}")


#--------------------------------------------------------------------------||--#
def bpy_to_vtk(obj_name) :
    obj = bpy.data.objects[obj_name]
    if obj is None:
        raise ValueError(f"Object '{obj_name}' not found in Blender.")
    
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj_name}' is not a mesh.")
    
    vertices = np.array([vert.co[:] for vert in obj.data.vertices])
    
    faces = []
    for poly in obj.data.polygons:
        faces.append(poly.vertices)
    
    vtk_points = vtk.vtkPoints()
    for vertex in vertices:
        vtk_points.InsertNextPoint(vertex)

    vtk_cells = vtk.vtkCellArray()
    for face in faces:
        vtk_cells.InsertNextCell(len(face)) 
        for vertex in face:
            vtk_cells.InsertCellPoint(vertex)

    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.SetPoints(vtk_points)
    vtk_polydata.SetPolys(vtk_cells)
    return vtk_polydata


#--------------------------------------------------------------------------||--#
def Vertices2PolyData(vertices) : 
    vtk_points = vtk.vtkPoints()
    vtk_array = numpy_support.numpy_to_vtk(vertices, deep=True)
    vtk_points.SetData(vtk_array)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(vtk_points)
    return pointsPolydata


def PolyData2Vertices(polydata):
    vtk_points = polydata.GetPoints()
    vtk_array = vtk_points.GetData()
    vertices = numpy_support.vtk_to_numpy(vtk_array)
    return vertices


#--------------------------------------------------------------------------||--#
def PointInsideObject(obj, pointsPolydata) :
    ## https://examples.vtk.org/site/Cxx/PolyData/PointInsideObject/
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInputData( pointsPolydata )
    selectEnclosedPoints.SetSurfaceData( obj )
    selectEnclosedPoints.Update()

    insideArray = selectEnclosedPoints.GetOutput().GetPointData().GetArray("SelectedPoints")
    inside_np = numpy_support.vtk_to_numpy(insideArray)
    #
    #for i, value in enumerate(inside_np) :
    #    print(f"Tuple {i}: ", end="")
    #    print("inside" if value == 1 else "outside")
    #
    return np.array(inside_np, dtype='bool') 


#--------------------------------------------------------------------------||--#
def GetOutline(vtkObj, faces, Tria=False) : 
  outline = vtk.vtkOutlineFilter()  
  outline.SetInputData( vtkObj )
  if faces :
    outline.GenerateFacesOn() 
  outline.Update()
  outline = outline.GetOutputDataObject(0) 

  if Tria : 
    tria = vtk.vtkTriangleFilter()
    tria.SetInputData( outline )
    tria.Update()
    outline = tria.GetOutputDataObject(0)
  return outline


#--------------------------------------------------------------------------||--#
def Transform(vtkObj, Translate=[0.0,0.0,0.0], Scale=[1.0,1.0,1.0], Rotate=[0.0,0.0,0.0,0.0]):
  transform = vtk.vtkTransform()
  transform.Translate( Translate )
  transform.Scale( Scale )   
  #transform.RotateY(0.0)
  transform.RotateWXYZ(Rotate[0], Rotate[1:]) 

 #tf = vtk.vtkTransformPolyDataFilter()
  tf = vtk.vtkTransformFilter()
  tf.SetInputData( vtkObj ) 
  tf.SetTransform( transform ) 
  tf.Update() 
  return tf.GetOutputDataObject(0)  


#--------------------------------------------------------------------------||--#
def GetOctree(vtkObject) :
  ## Cells are assigned to octree spatial regions based on the location of their centroids.
  octree = vtk.vtkOctreePointLocator()
  octree.SetDataSet(vtkObject);
  octree.BuildLocator(); #print( octree )
  return octree 


#--------------------------------------------------------------------------||--#
class CollisionDetectionA : 

    def __init__(self) : 
        return 


    def vertices_inside_obj(self, obj, vertices):
        ids = np.arange( vertices.shape[0] ) 

        inside_bbox = self.__vertices_inside_bbox__(obj, vertices)
        inside_bbox = ids[inside_bbox]
        on_bbox = vertices[inside_bbox]
        
        inside_obj = self.__vertices_inside_obj__(obj, on_bbox)
        inside_obj = inside_bbox[inside_obj]

        on_obj = vertices[inside_obj]
        self.inside_obj = inside_obj 
        return inside_obj


    def __vertices_inside_bbox__(self, obj, vertices) : 
        outline = GetOutline(obj, True) 
        return self.__vertices_inside_obj__(outline, vertices)


    def __vertices_inside_obj__(self, obj, vertices) : 
        verticesPolydata = Vertices2PolyData(vertices) 
        inside_bool = PointInsideObject(obj, verticesPolydata) 
        inside_indices = np.nonzero(inside_bool)[0]
        return inside_indices


#--------------------------------------------------------------------------||--#
class Cloud : 

    def __init__(self): 
        self.cloud = {} 
        return 

    def VerticesAdd(self, name, obj): 
        points = PolyData2Vertices(obj)
        self.cloud[name] = points
        return 


    def Create(self) : 
        arr = np.vstack( list(self.cloud.values()) ) 
        print(arr.shape) 
        return arr 



#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
def intersection_create(pairs, objs) :
    for name1, name2 in pairs:
        if name1 in objs and name2 in objs :
            print( name1, name2 )
            obj1 = objs.get(name1)
            obj2 = objs.get(name2)

            #obj1 = GetTriangles(obj1) 
            #obj2 = GetTriangles(obj2) 
            ## BooleanOperation1(obj1, obj2) 
            #filter1 = convert_to_output(obj1)
            #filter2 = convert_to_output(obj2)
            #BooleanOperation2(obj1, obj2) # :( 

    #polydata = vtk.vtkPolyData()
    #polydata.SetPoints(points)
    #polydata.SetLines(lines)
    #return polydata


def network_lines_create(pairs, positions):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    point_id_map = {}  

    for name, pos in positions.items():
        pid = points.InsertNextPoint(pos)
        point_id_map[name] = pid

    for obj1, obj2 in pairs:
        if obj1 in point_id_map and obj2 in point_id_map:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_id_map[obj1])
            line.GetPointIds().SetId(1, point_id_map[obj2])
            lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def GetOBBTree( polyData, level, maxLevel=5): 
    obbTree = vtk.vtkOBBTree();
    obbTree.SetDataSet(polyData);
    obbTree.SetMaxLevel(maxLevel);
    obbTree.BuildLocator();
    
    corner = np.zeros(3)
    max = np.zeros(3)
    mid  = np.zeros(3)  
    min  = np.zeros(3)
    size  = np.zeros(3)
    obbTree.ComputeOBB(polyData, corner, max, mid, min, size);
    print("[GetOBBTree] min, max:", min, max, )

    representation = vtk.vtkPolyData()  
    obbTree.GenerateRepresentation(level, representation)
    print("[GetOBBTree] nGetNumberOfPoints:", representation.GetNumberOfPoints(), 
            "GetNumberOfCells:", representation.GetNumberOfCells() )
    #return representation

    extract_edges = vtk.vtkExtractEdges()
    extract_edges.SetInputData(representation)
    extract_edges.Update()
    return extract_edges.GetOutput()


def flip_normals(polydata):
    reverse_sense = vtk.vtkReverseSense()
    reverse_sense.SetInputData(polydata)
    reverse_sense.ReverseNormalsOn()
    reverse_sense.ReverseCellsOn()
    reverse_sense.Update()
    return reverse_sense.GetOutput()


def GetTriangles( vtkObj ) :
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData( vtkObj )
    geometryFilter.Update()

    vtp  = geometryFilter.GetOutput();
    tria = vtk.vtkTriangleFilter()
    tria.SetInputData( geometryFilter.GetOutput() )
    ##tria.PassVertsOff()
    ##tria.PassLinesOff() 
    tria.Update()
    return vtp


def CollisionDetection(vtpA, vtpB, tap='\t') :
    matrix1 = vtk.vtkMatrix4x4()
    transform0 = vtk.vtkTransform()

    collide = vtk.vtkCollisionDetectionFilter()
    collide.SetInputData(0, vtpA )
    collide.SetTransform(0, transform0)
    collide.SetInputData(1, vtpB )

    collide.SetMatrix(1, matrix1)
    collide.SetBoxTolerance(0.0)
    collide.SetCellTolerance(0.0)
    collide.SetNumberOfCellsPerNode(2)

    ##collide.SetCollisionModeToAllContacts()
    ##collide.SetCollisionModeToHalfContacts()
    collide.SetCollisionModeToFirstContact()

    collide.GenerateScalarsOn()
    collide.Update();

    NumberOfContacts = collide.GetNumberOfContacts()
    #print("NumberOfContacts: %d" % NumberOfContacts)
    return NumberOfContacts 


def BBoxesTest(a, b) :
    """  
    1. (amin <= bmax)&&(amax >= bmin)

            bmin     bmax 
            |--------|

        |--------|
        amin     amax 

    """  
    amin = a[0]; amax = a[1]; assert(amin < amax)   
    bmin = b[0]; bmax = b[1]; assert(bmin < bmax)    
    overlapping = (amin <= bmax) and (amax >= bmin)
    return overlapping 


def OverlappingWithBBoxes(ClustersI, ClustersJ) : 
    ClustersIJ = {}  

    for Idi,Clusteri in ClustersI.items() :  
        bboxi = Clusteri.GetBounds(); # [xmin, xmax, ymin, ymax, zmin, zmax]
        for Idj,Clusterj in ClustersJ.items() :
            if Idi == Idj : 
                pass
            else :  
                bboxj = Clusterj.GetBounds(); # [xmin, xmax, ymin, ymax, zmin, zmax]
                overlapping_x = BBoxesTest(bboxi[0:2],bboxj[0:2])  
                overlapping_y = BBoxesTest(bboxi[2:4],bboxj[2:4])  
                overlapping_z = BBoxesTest(bboxi[4:6],bboxj[4:6])  
                #overlapping   = overlapping_x or  overlapping_y or  overlapping_z; ## ??  
                overlapping   = overlapping_x and overlapping_y and overlapping_z; ## ??   
                if overlapping  :
                    if ClustersIJ.get(Idi,None) : 
                        ClustersIJ[Idi].append(Idj) 
                    else : 
                        ClustersIJ[Idi] = [Idj]  
    return ClustersIJ  


def OverlappingWithClusters(ClustersI, ClustersJ, ClustersIJ) :
    ClustersIJ_New = {} 
    for Idi,Idsj in ClustersIJ.items() :
        Clusteri = ClustersI.get(Idi,None); assert(Clusteri) 
        Clusteri = GetTriangles(Clusteri)
        for Idj in Idsj :  
            if Idi == Idj : 
                pass
            else :  
                Clusterj = ClustersJ.get(Idj,None); assert(Clusterj)
                Clusterj = GetTriangles(Clusterj)   
                Clusterk = CollisionDetection(Clusteri,Clusterj,"\t  |_") ## [vtkExtractEnclosedPoints] 
                #Clusterk = vtk_tools.CollisionDetection(Clusteri,Clusterj,"\t  |_") ## [vtkExtractEnclosedPoints] 
                if Clusterk :
                    if ClustersIJ_New.get(Idi,None) : 
                        ClustersIJ_New[Idi].append(Idj) 
                    else : 
                        ClustersIJ_New[Idi] = [Idj]  
    return ClustersIJ_New


def GetCenterOfMassFilter( vtkObj ):
    CoM = None
    if vtkObj.GetNumberOfPoints() :  
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData( vtkObj );
        centerOfMassFilter.SetUseScalarsAsWeights(False);
        centerOfMassFilter.Update();
        CoM = np.array( centerOfMassFilter.GetCenter() ) 
    return CoM


def BooleanOperation1(vtpA, vtpB, tap='\t'):
    assert(vtpA)
    assert(vtpB)
    print("vtpA:%d/%d " % (vtpA.GetNumberOfPoints(),vtpA.GetNumberOfCells()) + "vtpB:%d/%d " % (vtpB.GetNumberOfPoints(),vtpB.GetNumberOfCells()) )

    booleanOperation = vtk.vtkBooleanOperationPolyDataFilter() 
    booleanOperation.SetInputData( 0, vtpA );
    booleanOperation.SetInputData( 1, vtpB );

    ##booleanOperation.SetOperationToDifference() 
    booleanOperation.SetOperationToIntersection()

    try :  
        booleanOperation.Update() # Segmentation fault: 11 !! 
    except : 
        pass 

    vtpC  = booleanOperation.GetOutput(); #print( vtpC )
    assert(vtpC)

    print("%s[BooleanOperation]" % tap + " NumberOfPoints: %d " % vtpC.GetNumberOfPoints() + "NumberOfCells: %d " % vtpC.GetNumberOfCells() )
    return vtpC 



def convert_to_output(polydata):
     # Convert vtkPolyData to vtkAlgorithmOutput
    polydata_filter = vtk.vtkCleanPolyData()
    polydata_filter.SetInputData(polydata)
    polydata_filter.Update()
    return polydata_filter


def BooleanOperation2(obj1, obj2, tap='\t') :
    filter1 = convert_to_output(obj1)
    filter2 = convert_to_output(obj2)
    #print( obj1 )
    print("obj1 : %d/%d " % (obj1.GetNumberOfPoints(),obj1.GetNumberOfCells()) )
    print("obj2 : %d/%d " % (obj2.GetNumberOfPoints(),obj2.GetNumberOfCells()) )

    operation = vtk.vtkBooleanOperationPolyDataFilter.VTK_INTERSECTION
    booleanOperation = vtk.vtkBooleanOperationPolyDataFilter() 
    booleanOperation.SetOperation(operation) 
    booleanOperation.SetInputConnection( 0, filter1.GetOutputPort() )
    booleanOperation.SetInputConnection( 1, filter2.GetOutputPort() )
    booleanOperation.Update() 

    print("BooleanOperation2:", booleanOperation.GetOutput().GetNumberOfPoints() )
    return booleanOperation


def BooleanOperation2_Test1() :
    x = 0.0 
    sphere1 = vtk.vtkSphereSource()
    sphere1.SetCenter(-0.15 + x, 0.0, 0.0);
    sphere1.Update()

    sphere2 = vtk.vtkSphereSource()
    sphere2.SetCenter( 0.15 + x, 0.0, 0.0);
    sphere2.Update()

    result = BooleanOperation2(sphere1.GetOutput(), sphere2.GetOutput()) 

    plotter = Plotter() 
    plotter.ActorAdd(sphere1.GetOutput(), colors.GetColor3d("Red"), 0.1) 
    plotter.ActorAdd(sphere2.GetOutput(), colors.GetColor3d("Blue"), 0.1) 
    plotter.ActorAdd(result.GetOutput(), colors.GetColor3d("Black"), 1.0) 
    plotter.Show() 
    exit()

#BooleanOperation2_Test1() 

class CollisionDetectionB : 

    def __init__(self) : 
        self.objs = {} 
        self.coms = {} 
        return 

    def ObjectAdd(self, name, obj) :
        self.objs[name] = obj  
        self.coms[name] = GetCenterOfMassFilter(obj)


    def Calculate(self) : 
        relationship = OverlappingWithBBoxes(self.objs, self.objs) 
        #print(relationship)

        relationship = OverlappingWithClusters(self.objs, self.objs, relationship)
        #print(relationship)

        self.NetworkCreate(relationship)
        return 


    def ObjetShow(self) : 
        ids = np.random.randint(0, len(color_names), size=len(self.objs))
        
        plotter = Plotter() 
        for i,(name,obj) in enumerate(self.objs.items()) : 
            color = color_names[ids[i]]
            color = colors.GetColor3d(color)
            outline = GetOutline(obj, False)
            plotter.ActorAdd(outline, color, 1.0) 
            plotter.ActorAdd(obj, color, 0.5, name) 

        plotter.Show() 
        return 


    def NetworkCreate(self, relationship) :
        nodes = [(key, value) for key, values in relationship.items() 
                    for value in values]
        self.nodes = nodes 

        intersection_create(self.nodes, self.objs) 
        #BooleanOperation(vtpA, vtpB) 
        return 
    

    def NetworkShow(self) :
        ids = np.random.randint(0, len(color_names), size=len(self.objs))
        
        plotter = Plotter() 
        for i,(name,obj) in enumerate(self.objs.items()) : 
            color = color_names[ids[i]]
            color = colors.GetColor3d(color)
            plotter.ActorAdd(obj, color, 0.125, name) 

            #outline = GetOutline(obj, False)
            #outline = GetOBBTree(obj, 0)
            #plotter.ActorAdd(outline, color, 1.0) 

        poly = network_lines_create(self.nodes, self.coms)
        color = colors.GetColor3d("Green")
        plotter.ActorAdd(poly, color, 1.0) 

        plotter.Show() 
        return 




#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
bpy_mesh1 = create_suzanne("Suzanne1")
vtk_mesh1 = bpy_to_vtk(bpy_mesh1.name)
vtk_mesh2 = Transform(vtk_mesh1, [1.5,1.5,0.0]) 
vtk_mesh3 = Transform(vtk_mesh1, [0.0,0.0,1.5]) 
vtk_mesh4 = Transform(vtk_mesh1, [2.0,2.0,0.0]) 

collisionDetectionB1 = CollisionDetectionB() 
collisionDetectionB1.ObjectAdd("obj1", vtk_mesh1) 
collisionDetectionB1.ObjectAdd("obj2", vtk_mesh2) 
collisionDetectionB1.ObjectAdd("obj3", vtk_mesh3) 
collisionDetectionB1.ObjectAdd("obj4", vtk_mesh4) 

#collisionDetectionB1.ObjetShow() 
collisionDetectionB1.Calculate() 
collisionDetectionB1.NetworkShow()


exit() 
## Plotting...
#points_all = Vertices2PolyData(vertices) 
#on_mesh1 = Vertices2PolyData(vertices[collisionDetection1.inside_obj]) 
"""
plotter = Plotter() 
plotter.VerticesAdd(points_all, (1.0,0.0,0.0)) 
plotter.VerticesAdd(on_mesh1, (0.0,1.0,0.0)) 

outline1 = GetOutline(vtk_mesh1, False)
plotter.ActorAdd(outline1, colors.GetColor3d("Cyan"), 1.0) 
plotter.ActorAdd(vtk_mesh1, colors.GetColor3d("Cyan"), 0.5, "obj1") 

outline2 = GetOutline(vtk_mesh2, False)
plotter.ActorAdd(outline2, colors.GetColor3d("RoyalBlue"), 1.0) 
plotter.ActorAdd(vtk_mesh2, colors.GetColor3d("RoyalBlue"), 0.5, "obj2") 

outline3 = GetOutline(vtk_mesh3, False)
plotter.ActorAdd(outline3, colors.GetColor3d("DarkGray"), 1.0) 
plotter.ActorAdd(vtk_mesh3, colors.GetColor3d("DarkGray"), 0.5, "obj3") 

outline4 = GetOutline(vtk_mesh4, False)
plotter.ActorAdd(outline4, colors.GetColor3d("Green"), 1.0) 
plotter.ActorAdd(vtk_mesh4, colors.GetColor3d("Green"), 0.5, "obj4") 


plotter.Show() 
"""

print("ok!")
#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#