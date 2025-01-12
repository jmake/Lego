import os
import sys
import re 
import numpy as np

import vtk
from vtk.util import numpy_support


#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
os.system("clear")
print("-----------------------")
print("+Running '%s'... " % sys.argv[0].split("/")[-1].split(".")[:-1][0] ) 

contr  = vtk.vtkMultiProcessController.GetGlobalController()
nranks = contr.GetNumberOfProcesses() if contr else 1
rank   = contr.GetLocalProcessId()    if contr else 0



GetVersion = lambda : vtk.vtkVersion.GetVTKSourceVersion()
getDicFromArray = lambda _A:{_u:np.nonzero(_A==_u)[0] for _u in np.unique(_A) }
Array2tuples = lambda nparray : list(map(tuple,nparray))

GetNumbersFromString = lambda _s : [ float(n) for n in re.findall('\d+', _s) ] 


#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
class RotateVector :
  def __init__(self,p1,p2,c0) :
    ## X.0.  
    p1 = np.array(p1); self.p1 = p1
    p2 = np.array(p2); self.p2 = p2
    c0 = np.array(c0); self.c0 = c0

    self.v1 = p1 - c0
    self.v2 = p2 - c0

    ## X.0.  
    self.axis,self.angle = self.Init(self.v1,self.v2)

    ## X.0.  
    self.m1 = np.linalg.norm(self.v1)
    self.m2 = np.linalg.norm(self.v2)
    self.u1 = self.v1 / self.m1
    self.u2 = self.v2 / self.m2
    return


  def Apply(self, angle_fraction, magnitud_fraction) : ## magnitud_fraction : 0 -> self.m1; 1 -> self.m2 
    from scipy.spatial.transform import Rotation as R
    #self.angle = np.pi 

    magnitude  = (self.m2/self.m1-1.0) * magnitud_fraction + 1.0  
#    magnitude  = (self.m2-self.m1) * magnitud_fraction + self.m1  
    angle      = self.angle * angle_fraction
    quaternion = self.GetQuaternion(self.axis,angle)
    Rotation   = R.from_quat( quaternion );
    u1_rot     = Rotation.apply( self.u1 )
#    if angle_fraction==1 :
#      ok = np.all( np.abs(u1_rot * self.m2 - self.v2) < 1e-3); assert(ok)
#      ok = np.all( np.abs(u1_rot * self.m2 + self.c0 - self.p2) < 1e-3); assert(ok)
#    if angle_fraction==0 :
#      ok = np.all( np.abs(u1_rot * self.m1 - self.v1) < 1e-3); assert(ok)
#      ok = np.all( np.abs(u1_rot * self.m1 + self.c0 - self.p1) < 1e-3); assert(ok)
    return u1_rot * self.m1 * magnitude + self.c0
#    return u1_rot * self.m1 * magnitud + self.c0
#    return u1_rot * magnitude + self.c0


  def GetMagnitudeRatio(self) : 
    return self.m2 / self.m1   


  def Init(self,v1,v2) :
    ## X.0.  
    cross = np.cross(v1,v2)
    cross = cross / np.linalg.norm(cross)
    assert( np.linalg.norm(cross) == 1.0)

    ## X.0.  
    angle = np.dot(v1,v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )
    angle = np.arccos(angle)
    #print( np.rad2deg(angle) )
    return cross, angle


  def GetQuaternion(self, U, phi) :
    q = [U[0]*np.sin(phi/2), U[1]*np.sin(phi/2), U[2]*np.sin(phi/2), np.cos(phi/2) ]
    q = np.array(q)
    q = q / np.linalg.norm( q )
    return q



#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
GetCoords    = lambda _ds : numpy_support.vtk_to_numpy( _ds.GetPoints().GetData() )
GetPointData = lambda _pd : {_pd.GetArrayName(k):numpy_support.vtk_to_numpy(_pd.GetArray(k)) for k in range(_pd.GetNumberOfArrays())}


#--------------------------------------------------------------------------||--#
def Threading() :
  """
  SEE : https://blog.kitware.com/simple-parallel-computing-with-vtksmptools-2/ 
  """
  return 


#--------------------------------------------------------------------------||--#
def VtkWarning( fname ) :  
  outwin = vtk.vtkFileOutputWindow();
  outwin.SetFileName( fname );
  outwin.SetInstance( outwin );
  return 


#--------------------------------------------------------------------------||--#
def CreateMovie(plotsName, movieName):
  SCRIPT = """ 
  mv MNAME MNAME_old

  ffmpeg -r 5 -pattern_type glob -i '*.png' -c:v libx264 -vf fps=100 -pix_fmt yuv420p out.mp4
 
  #ffmpeg -r 10 -i FNAME -c:v libx264 -vf fps=10 -pix_fmt yuv420p MNAME

  """
  SCRIPT = SCRIPT.replace("MNAME", movieName) 
  SCRIPT = SCRIPT.replace("FNAME", plotsName); print( SCRIPT )   
  os.system(SCRIPT) 
  print("\t[CreateMovie] '%s' Created! \n " % (movieName) ) 


#--------------------------------------------------------------------------||--#
def REPLACE_IN_FILE(fname, DIC, show=True) :
  fin = open(fname, "r")
  lines = fin.readlines()
  fin.close()

  for KEY,NEW in DIC.items() : 
    idx = [i for i,l in enumerate(lines) if KEY in l]; #print(idx)
    for i in idx : lines[i] = lines[i].replace(KEY,NEW)
    if show :
      for i in idx : print("\t line %d:'%s' changed! " % (i, lines[i].strip() ) )

  path,fin = os.path.split(fname)
  name,typ = os.path.splitext(fin); #print( path, fin)

  fout = "".join([path,name+"_changed"+typ]) ;  #print( fout )

  fin = open(fout, "w")
  fin.write( "".join(lines) )
  fin.close()
  return fin.name  


#--------------------------------------------------------------------------||--#
def Vtu2Vtp( vtkObj ) :
  assert( vtkObj.IsA("vtkUnstructuredGrid") )  
  geometryFilter = vtk.vtkGeometryFilter()  
  geometryFilter.SetInputData( vtkObj )  
  geometryFilter.Update()  
  return geometryFilter.GetOutputDataObject(0)   


def Vtx2Vtp( vtkObj ) :
#  assert( vtkObj.IsA("vtkUnstructuredGrid") )
  geometryFilter = vtk.vtkGeometryFilter()
  geometryFilter.SetInputData( vtkObj )
  geometryFilter.Update()
  return geometryFilter.GetOutputDataObject(0)


def GetResampleWithDataSet(target, source) : 
  r = vtk.vtkResampleWithDataSet() 
  r.SetSourceData( source )
  r.SetInputData( target )
  r.Update()
  return r.GetOutputDataObject(0)


def GetResampleToImage(target, dimensions, bounds) :
  try: 
    r = vtk.vtkPResampleToImage()
  except: 
    r = vtk.vtkResampleToImage() 
  #r.SetInputData( target )
  #r.SetSourceData( target )
  r.SetInputDataObject( target ) 
  r.SetSamplingDimensions( dimensions )
  r.SetSamplingBounds( bounds ) 
  r.Update()
  return r.GetOutputDataObject(0)


#--------------------------------------------------------------------------||--#    
def GetResampleWithShepardMethod( Obj, resolution) : 
  interpolator = vtk.vtkShepardMethod() 
  interpolator.SetInputData( Obj );
  interpolator.SetModelBounds( Obj.GetBounds() );
  interpolator.SetSampleDimensions(resolution, resolution, resolution);
  #interpolator.SetNullValue(-10000);
  interpolator.SetMaximumDistance( 0.005 ) 
  interpolator.Update();
  #print(  interpolator.GetMaximumDistance() )

  probe = vtk.vtkProbeFilter();
  probe.SetInputData(0, Obj);
  probe.SetInputConnection(1, interpolator.GetOutputPort() );
  probe.Update();
  #print( probe.GetOutputDataObject(0) )

  return interpolator.GetOutputDataObject(0)



#--------------------------------------------------------------------------||--#
def GetMassProperties1( vtkObj ) : 
  """
  Currently only triangles are processed. 
  Closed surface
  """
  assert( vtkObj.IsA("vtkPolyData") )

  mproperties = vtk.vtkMassProperties() 
  mproperties.SetInputData( vtkObj ) #geometryFilter.GetOutputDataObject(0)  );
  mproperties.Update();

  Vol  = mproperties.GetVolume ()
  Area = mproperties.GetSurfaceArea ()
  print("Area1,Vol1:", Area, Vol)

  return [Area,Vol]


#--------------------------------------------------------------------------||--#
def GetMassProperties2( vtkObj ) :  
  """
  An object is valid if each edge is used exactly two times by two different polygons. 
  Outputs :  "ObjectValidity", "ObjectVolumes" and "ObjectAreas" 
             "ObjectIds", "Areas" and "Volumes". 
             TotalVolume=sum(ObjectVolumes) 
             TotalArea=sum(ObjectAreas) 

  """
  assert( vtkObj.IsA("vtkPolyData") )  

  mproperties = vtk.vtkMultiObjectMassProperties()
  mproperties.SetInputData( vtkObj ) #.GetOutputDataObject(0)  );
  mproperties.Update();

  Vol  = mproperties.GetTotalVolume ()
  Area = mproperties.GetTotalArea ()
  print("Area2,Vol2:", Area, Vol)

  return 


#--------------------------------------------------------------------------||--#
def GetTriangles( vtkObj ) :
  geometryFilter = vtk.vtkGeometryFilter()
  geometryFilter.SetInputData( vtkObj )
  geometryFilter.Update()

  vtp  = geometryFilter.GetOutput();
  tria = vtk.vtkTriangleFilter()
  tria.SetInputData( geometryFilter.GetOutput() )
  #tria.PassVertsOff()
  #tria.PassLinesOff() 
  tria.Update()

  #vtp  = tria.GetOutput()
  return vtp


#--------------------------------------------------------------------------||--#
def Reader( fname ):
  #reader = vtk.vtkXMLGenericDataObjectReader()
  reader = None 
  if("vtu"  in fname) : reader = vtk.vtkXMLUnstructuredGridReader() 
  if("vtp"  in fname) : reader = vtk.vtkXMLPolyDataReader()
  if("pvd"  in fname) : reader = vtk.vtkPVDReader()  
  if("pvtp" in fname) : reader = vtk.vtkXMLPPolyDataReader()  
  if("pvtu" in fname) : reader = vtk.vtkXMLPUnstructuredGridReader()   
  if("vti"  in fname) : reader = vtk.vtkXMLImageDataReader() 
  if("vtm"  in fname) : reader = vtk.vtkXMLMultiBlockDataReader() 

  assert( reader )
  reader.SetFileName( fname );
  reader.UpdateInformation()
  reader.GetOutputInformation(0).Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_NUMBER_OF_PIECES(), nranks)
  reader.GetOutputInformation(0).Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_PIECE_NUMBER(), rank)
  reader.Update()

  NumberOfTimeSteps   = reader.GetNumberOfTimeSteps();
  NumberOfPointArrays = reader.GetNumberOfPointArrays();

  dataObject     = reader.GetOutputDataObject(0);    
  ClassName      = dataObject.GetClassName() 
  print("\t[Reader] File:'%s' ClassName:'%s' " % (fname, ClassName) )

  try : 
    NumberOfPoints = dataObject.GetNumberOfPoints() 
    NumberOfCells  = dataObject.GetNumberOfCells()
    print("\t         NumberOfPoints:%d NumberOfCells:%d NumberOfPointArrays:%d " % (NumberOfPoints, NumberOfCells, NumberOfPointArrays) )
  except : 
    pass 

  #vtu = reader.GetOutputAsDataSet() if reader.IsA("vtkPVDReader") else reader.GetOutput() 
  #vtu = dataObject if dataObject.IsA("vtkUnstructuredGrid") else None  
  #vtp = GetTriangles( vtu ) 
  return dataObject  


#--------------------------------------------------------------------------||--#
def Writer( vtkObj, fname ):
  writer    = None 
  extension = None 

  if vtkObj.IsA("vtkPolyData"         ) : writer = vtk.vtkXMLPolyDataWriter()  
  if vtkObj.IsA("vtkUnstructuredGrid" ) : writer = vtk.vtkXMLUnstructuredGridWriter()
  if vtkObj.IsA("vtkMultiBlockDataSet") : writer = vtk.vtkXMLMultiBlockDataWriter()
  if vtkObj.IsA("vtkHyperTreeGrid"    ) : writer = vtk.vtkXMLHyperTreeGridWriter()
  if vtkObj.IsA("vtkRectilinearGrid"  ) : writer = vtk.vtkXMLRectilinearGridWriter() 
  if vtkObj.IsA("vtkImageData"        ) : writer = vtk.vtkXMLImageDataWriter()
  #if vtkObj.IsA(""  ) : writer = vtk.vtkXML Writer()

  # DATA_TIME_STEP
  try :
    writer.SetFileName( "%s.%s" % (fname,writer.GetDefaultFileExtension()) )
    writer.SetInputData( vtkObj )
    writer.Write()
    print("\t[Writer] '%s' Saved!!" %(writer.GetFileName()) )
  except : 
    print("\t[Writer] Class '%s' Not Saved!!" %( vtkObj.GetClassName() ) )
    exit(0)  

  return 


#--------------------------------------------------------------------------||--#
def BooleanOperation(vtpA, vtpB, tap='\t'):
  assert(vtpA)
  assert(vtpB)

  print("vtpA:%d/%d " % (vtpA.GetNumberOfPoints(),vtpA.GetNumberOfCells()) + "vtpB:%d/%d " % (vtpB.GetNumberOfPoints(),vtpB.GetNumberOfCells()) )

  booleanOperation = vtk.vtkBooleanOperationPolyDataFilter() 

#  booleanOperation.SetOperationToIntersection()
  booleanOperation.SetOperationToDifference() 

  booleanOperation.SetInputData( 0, vtpA );
  booleanOperation.SetInputData( 1, vtpB );

#  Writer(vtpA, "a")
#  Writer(vtpB, "b")

  try :  
    booleanOperation.Update() # Segmentation fault: 11 !! 
  except : 
    pass 

  vtpC  = booleanOperation.GetOutput(); #print( vtpC )
  assert(vtpC)

  print("%s[BooleanOperation]" % tap + " NumberOfPoints: %d " % vtpC.GetNumberOfPoints() + "NumberOfCells: %d " % vtpC.GetNumberOfCells() )
  return vtpC 

 
#--------------------------------------------------------------------------||--#
def FindCells2(vtkObject, boundingBox):
  FoundCells  = vtk.vtkIdList()

##cellLocator = vtk.vtkCellLocator() # :( Doesnt work ... 
  cellLocator = vtk.vtkCellTreeLocator()
  cellLocator.SetDataSet( vtkObject )

  cellLocator.AutomaticOn()
  cellLocator.SetNumberOfCellsPerNode(20)
  cellLocator.CacheCellBoundsOn()
  cellLocator.BuildLocator()

  cellLocator.FindCellsWithinBounds(boundingBox,FoundCells)
  FoundCells = np.array([ FoundCells.GetId(i) for i in range(FoundCells.GetNumberOfIds()) ])
  print("vtkCellTreeLocator.nFoundCells:", FoundCells.shape[0] )

  nArray = vtkObject.GetNumberOfCells() 
  Array = vtk.vtkFloatArray()
  Array.SetName("Touched")
  Array.SetNumberOfComponents(1)
  Array.SetNumberOfTuples(nArray)
  for i in range(nArray): Array.SetValue(i,0)
  for i in FoundCells: Array.SetValue(i,1)

  vtkObject.GetCellData().AddArray(Array) 

  maxLevel = cellLocator.GetMaxLevel(); print("vtkCellTreeLocator.maxLevel:", maxLevel)
  representation = vtk.vtkPolyData()
  cellLocator.GenerateRepresentation(maxLevel, representation)

  return vtkObject, representation

#--------------------------------------------------------------------------||--#
def GetOctree(vtkObject):
  ## Cells are assigned to octree spatial regions based on the location of their centroids.
  octree = vtk.vtkOctreePointLocator()
  octree.SetDataSet(vtkObject);
  octree.BuildLocator(); #print( octree )
  return octree 


#--------------------------------------------------------------------------||--#
def GetKdtree(vtkObject):
  vtkpoints = vtkObject.GetPoints(); #print( vtkpoints )

  octree = vtk.vtkPKdTree()
  octree.SetDataSet(vtkObject);
  octree.BuildLocatorFromPoints( vtkpoints ); print( octree )  

  assert( octree )
  return octree


#--------------------------------------------------------------------------||--#
def FindCells3(vtkObject, boundingBox):
  octree = GetOctree(vtkObject)
  nPtsInTree = octree.GetDataSet().GetNumberOfPoints()
  #print("Number of points in tree: ", nPtsInTree)

  FoundPts = vtk.vtkIdTypeArray()
  octree.FindPointsInArea(boundingBox,FoundPts)
  FoundPts = numpy_support.vtk_to_numpy( FoundPts )
  print("\t[vtkOctreePointLocator] nFoundPts: %d" % FoundPts.shape[0] )

  if FoundPts.shape[0] : 
    nArray = vtkObject.GetNumberOfPoints() 
    Array = vtk.vtkFloatArray()
    Array.SetName("Touched")
    Array.SetNumberOfComponents(1)
    Array.SetNumberOfTuples(nArray)
    for i in range(nArray): Array.SetValue(i,0)
    for i in FoundPts: Array.SetValue(i,1)
    vtkObject.GetPointData().AddArray(Array) 

  maxLevel = octree.GetMaxLevel(); print("\t[vtkOctreePointLocator]  maxLevel: %d" % maxLevel)
  vtp      = vtk.vtkPolyData()
  octree.GenerateRepresentation(maxLevel, vtp)

  return vtkObject, vtp


#--------------------------------------------------------------------------||--#
def Transform(vtkObj, Translate=[0.0,0.0,0.0], Scale=[1.0,1.0,1.0]):
  transform = vtk.vtkTransform()
  transform.Translate( Translate )
  transform.Scale( Scale )   
  transform.RotateY(0.0)

 #tf = vtk.vtkTransformPolyDataFilter()
  tf = vtk.vtkTransformFilter()
  tf.SetInputData( vtkObj ) 
  tf.SetTransform( transform ) 
  tf.Update() 
  return tf.GetOutputDataObject(0)  


def GetNormalized( vtkObj ): 
  bbox    = np.array(vtkObj.GetBounds()).reshape( (3,2) )
  mins    = bbox[:,0]
  dbbox   = 1.0/(bbox[:,1]-mins)
  vtkObj  = Transform(vtkObj, Translate=-mins  ) 
  vtkObj  = Transform(vtkObj,     Scale= dbbox ) 
  return vtkObj 


#--------------------------------------------------------------------------||--#
def ExtractCluster(vtkObj, IDs, Using):
  ids = vtk.vtkIdTypeArray()
  ids.SetNumberOfComponents(1)
  for id in IDs: ids.InsertNextValue(id)

  selectionNode = vtk.vtkSelectionNode()
##selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT);
##selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL);
  if Using=="CELL" :  selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL);
  if Using=="POINT" : selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT);

  selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES);
  selectionNode.SetSelectionList(ids);

  selection = vtk.vtkSelection()
  selection.AddNode(selectionNode);

  extractSelection = vtk.vtkExtractSelection()
  extractSelection.SetInputData(0, vtkObj);
  extractSelection.SetInputData(1, selection);
  extractSelection.Update();
#
#  selected = vtk.vtkUnstructuredGrid()
#  #selected = vtk.vtkPolyData()  
#  selected.DeepCopy( extractSelection.GetOutput() )
#  #print("nGetNumberOfPoints:", selected.GetNumberOfPoints(), 
#  #        "nGetNumberOfCells:", selected.GetNumberOfCells() )
#  return selected
#
  return extractSelection.GetOutputDataObject(0)
 

#--------------------------------------------------------------------------||--#
def GetExtractGeometry( vtkObj, bbox ) :
  box = vtk.vtkBox() 
  box.SetBounds(bbox)   
 
  eg = vtk.vtkExtractGeometry() 
  eg.SetInputData( vtkObj )  
  eg.SetImplicitFunction( box )  
  eg.ExtractInsideOn()
  eg.ExtractBoundaryCellsOn()
  eg.Update()
  return eg.GetOutputDataObject(0)   


#--------------------------------------------------------------------------||--#
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

  return representation


#--------------------------------------------------------------------------||--#
def __get_cell_data__(GetOutput, property_name=None):
    Data    = GetOutput.GetCellData() 
    n_props = Data.GetNumberOfArrays()
    #for i in range(n_props):
    #  print " |_%d) \\\\\\\'%s\\\\\\\'" % (i+1, Data.GetArrayName(i) )

    DATA = None
    if(property_name!=None):
      #print "[get_cell_data]", 
      Prop   = Data.GetArray(property_name)
      n_cols = Prop.GetNumberOfComponents()
      n_rows = Prop.GetNumberOfTuples() 
      #print "\\\\\\\'%s\\\\\\\':" % Prop.GetName(),   

      prop_range = Prop.GetRange()   
      #print "[%f,%f]" %( prop_range[0], prop_range[1] )

      k=0
      DATA = []
      if(n_cols>1): 
        for j in range(n_rows): 
          data = []
          for i in range(n_cols): 
            data.append( Prop.GetValue(k) ) 
            k+=1
          DATA.append( data ) 
    
      if(n_cols==1): 
        for j in range(n_rows): 
          for i in range(n_cols): 
            DATA.append( Prop.GetValue(k) ) 
            k+=1
    
    return np.array(DATA)


#--------------------------------------------------------------------------||--#
def CheckArrayName(vtkObj, key=None): 
  pointData = vtkObj.GetPointData()
  NumberOfArrays = pointData.GetNumberOfArrays()
  ArraysName = [pointData.GetArrayName(iarray) for iarray in range(NumberOfArrays)]

  ok = key in ArraysName 
  if not ok :
    print( "\t[CheckArrayName] '%s' not found!! \n\t[CheckArrayName] Use:" % key )
    for iarray in range(NumberOfArrays): print( "\t\t'%s' " % pointData.GetArrayName(iarray) )
    exit(0)
 
  return ok 


#--------------------------------------------------------------------------||--#
def GetPointDataArray(vtkObj, key=None): 
  with np.printoptions(precision=3, suppress=True):
    Array     = np.zeros(0) 
    pointData = vtkObj.GetPointData()  

    NumberOfArrays = pointData.GetNumberOfArrays() 
    ArraysName = [pointData.GetArrayName(iarray) for iarray in range(NumberOfArrays)] 

    if key in ArraysName  : 
      Array = pointData.GetArray(key)
      Array = numpy_support.vtk_to_numpy(Array)
    else :
      print( "\t[GetDataArray] '%s' not found!! \n\t[GetDataArray] Use:" % key ) 
      for iarray in range(NumberOfArrays): print( "\t\t'%s' " % pointData.GetArrayName(iarray) )
      exit(0)
    return Array  


#--------------------------------------------------------------------------||--#
def GetCellDataArray(vtkObj, key=None):
  with np.printoptions(precision=3, suppress=True):
    Array    = np.zeros(0)
    cellData = vtkObj.GetCellData()

    NumberOfArrays = cellData.GetNumberOfArrays()
    ArraysName = [cellData.GetArrayName(iarray) for iarray in range(NumberOfArrays)]

    if key in ArraysName :
      Array = cellData.GetArray(key)
      Array = numpy_support.vtk_to_numpy(Array)
    else :
      print( "\t[GetDataArray] '%s' not found!! \n\t[GetDataArray] Use:" % key )
      for iarray in range(NumberOfArrays): print( "\t\t'%s' " % cellData.GetArrayName(iarray) )
      exit(0)
    return Array


#--------------------------------------------------------------------------||--#
def GetFieldDataArray(vtkObj, key=None):
  with np.printoptions(precision=3, suppress=True):
    Array    = np.zeros(0)
    cellData = vtkObj.GetFieldData()

    NumberOfArrays = cellData.GetNumberOfArrays()
    ArraysName = [cellData.GetArrayName(iarray) for iarray in range(NumberOfArrays)]

    if key in ArraysName :
      Array = cellData.GetArray(key)
      Array = numpy_support.vtk_to_numpy(Array)
    else :
      print( "\t[GetDataArray] '%s' not found!! \n\t[GetDataArray] Use:" % key )
      for iarray in range(NumberOfArrays): print( "\t\t'%s' " % cellData.GetArrayName(iarray) )
      exit(0)
    return Array


#--------------------------------------------------------------------------||--#
def SetFieldData(vtkObj, array, key) : 
  fieldData = vtkObj.GetFieldData(); #NumberOfArrays = fieldData.GetNumberOfArrays()
  vtkarray = numpy_support.numpy_to_vtk(array) 
  vtkarray.SetName(key)  
  fieldData.AddArray( vtkarray )
  return vtkObj #vtkarray 

 
def GetVtkArray(array, key, ndim=1) :
  vtkarray = numpy_support.numpy_to_vtk(array) #, array_type=ndim)
  vtkarray.SetName(key)   
  vtkarray.SetNumberOfComponents(ndim)  
  return vtkarray


#--------------------------------------------------------------------------||--#
def SetIdentifiers(vtkObj, key) :  
  idFilter = vtk.vtkIdFilter() 
  idFilter.SetInputData( vtkObj );
  idFilter.SetIdsArrayName(key); # Paraview 5.6.0 
  #idFilter.SetPointIdsArrayName(key) # Paraview 5.7.0  
  idFilter.Update();
  return idFilter.GetOutputDataObject(0)


#--------------------------------------------------------------------------||--#
def GetClusters( vtkObj, key, threshold): 
  contourFilter = vtk.vtkContourFilter();
  #contourFilter.SetInputConnection( reader.GetOutputPort() );
  contourFilter.SetInputData( vtkObj );
  contourFilter.SetValue(0,threshold); # IsoSurface(0)= threshold  
  contourFilter.SetInputArrayToProcess( 0, 0, 0, 0, key)
  contourFilter.Update();
  #print( contourFilter.GetOutputPort().GetClassName() )
  #print( contourFilter.GetOutputDataObject(0).GetClassName() )

  vtkObj = contourFilter.GetOutputDataObject(0) # .GetClassName() 
  vtkPointData = vtkObj.GetPointData()
  #print( [vtkPointData.GetArrayName(i) for i in range(vtkPointData.GetNumberOfArrays())] )

  connectivityFilter = vtk.vtkConnectivityFilter()
  connectivityFilter.SetInputConnection( contourFilter.GetOutputPort() );
  connectivityFilter.SetExtractionModeToAllRegions();
  connectivityFilter.ColorRegionsOn();
  connectivityFilter.Update();
  print("\t[GetConnectivity] IDs: %d" % connectivityFilter.GetNumberOfExtractedRegions() )

  #vtkObj = connectivityFilter.GetOutputDataObject(0) # .GetClassName() 
  #vtkPointData = vtkObj.GetPointData()
  #print( [vtkPointData.GetArrayName(i) for i in range(vtkPointData.GetNumberOfArrays())] )
  #return vtkObj  
  return connectivityFilter.GetOutputDataObject(0) 


def GetThreshold(vtkObj, key, lower, upper) : 
  t = vtk.vtkThreshold()  
  t.SetInputData( vtkObj )
  #t.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, key);
  #t.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, key);

  if( (lower != None) and (upper == None) ) : 
    t.ThresholdByLower(lower) #  less or equal 
  if( (lower == None) and (upper != None) ) :
    t.ThresholdByUpper(upper) #   greater or equal  
  if( (lower != None) and (upper != None) ) :
    t.ThresholdBetween(lower,upper)

  t.SetInputArrayToProcess( 0, 0, 0, 0, key)
  #t.SetAttributeModeToUsePointData () # This method is deprecated, please use SetInputArrayToProcess instead. 
  #print( "\t[GetThreshold] Attribute:'%s' " % t.GetAttributeModeAsString() ) 

  t.Update()
  return t.GetOutputDataObject(0)


def GetCountour(vtkObj, key, threshold):
  contourFilter = vtk.vtkContourFilter();
  contourFilter.SetInputData( vtkObj );
  contourFilter.SetValue(0,threshold); # IsoSurface(0)= threshold  
  contourFilter.SetInputArrayToProcess( 0, 0, 0, 0, key)
  contourFilter.Update();
  return contourFilter.GetOutputDataObject(0)



def GetConnectivity(vtkObj) : #, key): 
  try :  
    connectivityFilter = vtk.vtkPConnectivityFilter()
  except: 
    connectivityFilter = vtk.vtkConnectivityFilter()
  connectivityFilter.SetInputData( vtkObj )
  #connectivityFilter.SetInputArrayToProcess( 0, 0, 0, 0, key)
  connectivityFilter.SetExtractionModeToAllRegions();
  connectivityFilter.ColorRegionsOn();
  connectivityFilter.Update();

  print("\t[GetConnectivity] IDs: %d" % connectivityFilter.GetNumberOfExtractedRegions() )
  return connectivityFilter.GetOutputDataObject(0)


def GetGeometry(vtkObj) :  
  #g = vtk.vtkGeometryFilter() 
  g = vtk.vtkExtractGrid()  
  g.SetInputData( vtkObj )
  g.Update();
  return g.GetOutputDataObject(0)



#--------------------------------------------------------------------------||--#
def GetOutline( vtkObj, Tria=False) : 
#  try :  
#    outline = vtk.vtkPOutlineFilter()
#  except : 
#    outline = vtk.vtkOutlineFilter()  
  outline = vtk.vtkOutlineFilter()  
  outline.SetInputData( vtkObj )
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
def AppendVtps( vtkObjList ) : 
  appendPolyData = vtk.vtkAppendPolyData() 

  for vtkObj in vtkObjList : appendPolyData.AddInputData(vtkObj)  
  appendPolyData.Update() 

  cleaned = vtk.vtkCleanPolyData()         
  cleaned.SetInputConnection(appendPolyData.GetOutputPort());
  cleaned.Update() 
 
  return cleaned.GetOutputDataObject(0)  


#--------------------------------------------------------------------------||--#
def AppendVtus( vtkObjList ):
  append = vtk.vtkAppendFilter() 
  for vtkObj in vtkObjList : append.AddInputData(vtkObj)
  append.Update()
  return append.GetOutputDataObject(0)  


#--------------------------------------------------------------------------||--#
def GetMultiBlockDataSetFromList( vtkObjList ):
  mb = vtk.vtkMultiBlockDataSet() 
  for ivtkObj,vtkObj in enumerate(vtkObjList):  mb.SetBlock(ivtkObj,vtkObj)
  return mb # -> vtm  


def GetMultiBlockDataSet( vtkObjDict ):
  mb = vtk.vtkMultiBlockDataSet()
  for k,v in vtkObjDict.items():  
    #v = SetFieldData(v,np.array([k]),"PathId")
    mb.SetBlock(k,v)
  return mb # -> vtm 


def GetTemporalDataSet( vtkObjDict ) :
  """
  AttributeError: module 'vtk' has no attribute 'vtkTemporalDataSet'
  """  
  c = vtk.vtkTemporalDataSet() 
  for k,v in vtkObjDict.items(): c.SetTimeStep(k,v)  
  return c  



#--------------------------------------------------------------------------||--#
def GetBlocks( vtm, block=None ):
#  b = vtk.vtkExtractBlock()
#vtkCompositeDataIterator
  """ 
vtkCompositeDataIterator* iter = compositeData->NewIterator();
for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
  vtkDataObject* dObj = iter->GetCurrentDataObject();
  cout << dObj->GetClassName() <<endl;
  }
  """ 

  iBlock = 0 
  Blocks = {} 
 
  Iter   = vtm.NewIterator() 
  Iter.InitTraversal();  
  while not Iter.IsDoneWithTraversal() :
    Blocks[iBlock] = Iter.GetCurrentDataObject()  
    Iter.GoToNextItem()
    iBlock += 1 

  return Blocks 


#--------------------------------------------------------------------------||--#
def GetCenterOfMassFilter( vtkObj ):
  CoM = None
  if vtkObj.GetNumberOfPoints() :  
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData( vtkObj );
    centerOfMassFilter.SetUseScalarsAsWeights(False);
    centerOfMassFilter.Update();
    CoM = np.array( centerOfMassFilter.GetCenter() ) 
  return CoM 



#--------------------------------------------------------------------------||--#
def GetArcLength( vtkObj ):
    a = vtk.vtkAppendArcLength()
    a.SetInputData( vtkObj ) 
    a.Update()
    return a.GetOutputDataObject(0)


#--------------------------------------------------------------------------||--#
def vtkMultiPieceDataSet(): 
  """
For example, say that a simulation broke a volume into 16 piece so that each piece can be processed with 1 process in parallel. We want to load this volume in a visualization cluster of 4 nodes. Each node will get 4 pieces, not necessarily forming a whole rectangular piece. 
  """ 
  return 


#--------------------------------------------------------------------------||--#
def DistancePolyDataFilter(  vtkObjA, vtkObjB ) : 
  distanceFilter =  vtk.vtkDistancePolyDataFilter();
##distanceFilter.SetInputConnection( 0, clean1->GetOutputPort() );
##distanceFilter.SetInputConnection( 1, clean2->GetOutputPort() );
  #
  distanceFilter.SetInputDataObject(0, vtkObjA );
  distanceFilter.SetInputDataObject(1, vtkObjB );
  #
  distanceFilter.Update();

  return distanceFilter.GetOutputDataObject(0)



#--------------------------------------------------------------------------||--#
def IntersectionPolyDataFilter( vtkObjA, vtkObjB ) : 
  ##
  """ 
  [Reader] File:'a.vtp' ClassName:'vtkPolyData'  NumberOfPoints:51 NumberOfCells:92 NumberOfPointArrays:7 
  [Reader] File:'b.vtp' ClassName:'vtkPolyData'  NumberOfPoints:49 NumberOfCells:88 NumberOfPointArrays:7 
  """ 
  ## 
  """ 
  Intersection = vtk.vtkIntersectionPolyDataFilter() 
##Intersection.SetInputConnection(0, this->GetInputConnection(0, 0));
##Intersection.SetInputConnection(1, this->GetInputConnection(1, 0));
  Intersection.SetInputDataObject(0, vtkObjA ); 
  Intersection.SetInputDataObject(1, vtkObjB ); 
#  Intersection.SplitFirstOutputOn(); 
#  Intersection.SplitSecondOutputOn(); 
  Intersection.Update();

  assert(Intersection.GetStatus() != 1) 
  """
  ##  
  """
  + vtkIntersectionPolyDataFilter.cxx 

  |_ RequestData(...)                       2363  
    |_ FindTriangleIntersections(...)       2496 
    |_ impl->SplitMesh(...)                 2558  
    |_ impl->SplitMesh(...)                 2598   

  |_ SplitMesh(...)                         0494  
     vtkCell *cell = input->GetCell(cellId) 0548 
  """
  ##
  """

 537 splitLines->GetNumberOfPoints() ## error: Execution was interrupted, reason: EXC_BAD_ACCESS (code=1, address=0x40).


2338 int vtkIntersectionPolyDataFilter::RequestData(... inputVector ... outputVector)  

2345   vtkInformation *outIntersectionInfo = outputVector->GetInformationObject(0);

2366   vtkPolyData     *outputIntersection = vtkPolyData::SafeDownCast( outIntersectionInfo->Get(vtkDataObject::DATA_OBJECT()) );

2511   this->NumberOfIntersectionPoints = outputIntersection->GetNumberOfPoints(); ## error: Execution was interrupted, reason: EXC_BAD_ACCESS (code=1, address=0x40).


2533     if (impl->SplitMesh(0, outputPolyData0, outputIntersection) != 1)
                   \__> 537 

2379   vtkSmartPointer< vtkPolyData > mesh0 = vtkSmartPointer< vtkPolyData >::New(); ## mesh0->GetPoints()->GetNumberOfPoints() = 51 

2382   vtkSmartPointer< vtkPolyData > mesh1 = vtkSmartPointer< vtkPolyData >::New(); ## mesh1->GetPoints()->GetNumberOfPoints() = 49   



  """


  """
  ::GetNumberOfElements(int type)   
  Returns the number of elements of the given type where type can vtkDataObject::POINT, vtkDataObject::CELL, ... 
  """

  return 


#--------------------------------------------------------------------------||--#
def CleanPolyData(  vtkObj ) : 
  clean1 =  vtk.vtkCleanPolyData();
  clean1.SetInputData( vtkObj );
  clean1.Update();

  return clean1.GetOutputDataObject(0)


#--------------------------------------------------------------------------||--#
def GetMassProperties3( Obj ):
  geometryFilter = vtk.vtkGeometryFilter()
  geometryFilter.SetInputData( Obj )
  geometryFilter.Update()

  tria = vtk.vtkTriangleFilter()
  tria.SetInputData( geometryFilter.GetOutput() )
  tria.Update()

  Vol  = None 
  Area = None 
  if 1 : 
    mproperties1 = vtk.vtkMassProperties()
    mproperties1.SetInputData( tria.GetOutputDataObject(0) );
    mproperties1.Update();

    Vol = mproperties1.GetVolume ()
    Area = mproperties1.GetSurfaceArea ()
    #print("Area1,Vol1:", Area, Vol)

  if 0 :  
    mproperties2 = vtk.vtkMultiObjectMassProperties()
    mproperties2.SetInputData( tria.GetOutputDataObject(0)  );
    mproperties2.Update();

    Vol = mproperties2.GetTotalVolume ()
    Area = mproperties2.GetTotalArea ()
    print("Area2,Vol2:", Area, Vol)

  return [Area,Vol]




#--------------------------------------------------------------------------||--#
def CollisionDetection(vtpA, vtpB, tap='\t'):  
    #Writer(vtpA, "a")
    #Writer(vtpB, "b")

    matrix1 = vtk.vtkMatrix4x4()
    transform0 = vtk.vtkTransform()

    collide = vtk.vtkCollisionDetectionFilter()
    #collide.SetInputConnection(0, sphere0.GetOutputPort())
    collide.SetInputData(0, vtpA );
    collide.SetTransform(0, transform0)
    #collide.SetInputConnection(1, sphere1.GetOutputPort())
    collide.SetInputData(1, vtpB );

    collide.SetMatrix(1, matrix1)
    collide.SetBoxTolerance(0.0)
    collide.SetCellTolerance(0.0)
    collide.SetNumberOfCellsPerNode(2)
    #
    #collide.SetCollisionModeToAllContacts()
    collide.SetCollisionModeToFirstContact()
    #collide.SetCollisionModeToHalfContacts()
    #

    collide.GenerateScalarsOn()
    collide.Update();

    NumberOfContacts = collide.GetNumberOfContacts()
    #print("NumberOfContacts: %d" % NumberOfContacts)
    return NumberOfContacts 



#--------------------------------------------------------------------------||--#
def SmoothFilter( Obj ) : 
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData( Obj );
    smoothFilter.SetNumberOfIterations(50);
    smoothFilter.SetRelaxationFactor(0.2);
    smoothFilter.FeatureEdgeSmoothingOff();
    smoothFilter.BoundarySmoothingOn();
    smoothFilter.Update(); 
    return smoothFilter.GetOutputDataObject(0) 


#--------------------------------------------------------------------------||--#
def GetSignedDistances( Obj, Grid ) :  
  assert( Obj.IsA("vtkPolyData") )

  signedDistances = vtk.vtkFloatArray();
  signedDistances.SetNumberOfComponents(1);
  signedDistances.SetName("SignedDistances");

  implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance();
  implicitPolyDataDistance.SetInput( Obj );  # Set the input vtkPolyData used for the implicit function evaluation.  

  for pointId in range(Grid.GetNumberOfPoints()):
    p = [0.0,0.0,0.0];
    Grid.GetPoint(pointId, p);
    signedDistance = implicitPolyDataDistance.EvaluateFunction(p);
    signedDistances.InsertNextValue(signedDistance);
  #Grid.GetPointData().SetScalars(signedDistances);
  return signedDistances #.GetOutputDataObject(0)


#--------------------------------------------------------------------------||--#
def CreateRectilinearGrid(X, Y, Z) : 
  xCoords = vtk.vtkFloatArray();
  for x in X : xCoords.InsertNextValue(x);

  yCoords = vtk.vtkFloatArray();
  for y in Y : yCoords.InsertNextValue(y);

  zCoords = vtk.vtkFloatArray();
  for z in Z : zCoords.InsertNextValue(z);

  rgrid = vtk.vtkRectilinearGrid();
  rgrid.SetDimensions(xCoords.GetNumberOfTuples(),
                      yCoords.GetNumberOfTuples(),
                      zCoords.GetNumberOfTuples());
  rgrid.SetXCoordinates(xCoords);
  rgrid.SetYCoordinates(yCoords);
  rgrid.SetZCoordinates(zCoords);

#  print( rgrid ); exit(0)
  return rgrid 


#--------------------------------------------------------------------------||--#
def GetImageData( Extent ) :
  image  = vtk.vtkImageData() 
  image.SetExtent(Extent)  
  image.SetDimensions(10,10,1); 
  return image  


#--------------------------------------------------------------------------||--#
def GetClipper( Obj, value) : 
  """
  https://lorensen.github.io/VTKExamples/site/Cxx/Meshes/ClipDataSetWithPolyData/ 
  """ 
  clipper = vtk.vtkClipDataSet();
  clipper.SetInputData( Obj );
  clipper.SetValue( value );
  clipper.InsideOutOn();
  clipper.GenerateClippedOutputOn();
  clipper.CreateDefaultLocator()     ##  used to merge coincident points
  clipper.Update();

  inside  = clipper.GetOutputDataObject(0); #print( obj )
  outside = clipper.GetOutputDataObject(1); #print( obj )
  return [inside,outside]



def GetCourseMesh(Obj, reduction):
  decimate = vtk.vtkDecimatePro()
  decimate.SetInputData( Obj )   
  decimate.SetTargetReduction( reduction )
  decimate.PreserveTopologyOn()
  decimate.Update()

  return decimate.GetOutputDataObject(0) 


def GetMeshQuality( Obj ) :  
  quality = vtk.vtkMeshQuality()   
  quality.SetInputData( Obj )  
  quality.SetHexQualityMeasureToVolume() 
  quality.Update()  
  assert( vtk.VTK_QUALITY_VOLUME==quality.GetHexQualityMeasure() )   

  Quality = GetCellDataArray(quality.GetOutputDataObject(0), "Quality") 
  print( Quality ) 
 
  return quality.GetOutputDataObject(0)  



def GetCellsVolume( Obj ) : 
  """
  Default names are VertexCount, Length, Area and Volume. 
  """ 
  s = vtk.vtkCellSizeFilter()
  s.SetInputData( Obj )
  s.ComputeSumOn() 
  s.Update() 
  output = s.GetOutputDataObject(0); #print( output ) 

  name    = s.GetVolumeArrayName()   
  Volume1 = GetCellDataArray(output,name); assert( np.all(Volume1 > 0.0) )
  Volume1 = np.sum(Volume1) ; #print( name, Volume1 )
  Volume2 = GetFieldDataArray(output,'Volume')[0]; #print( name, Volume2 )

  assert( np.abs(Volume1-Volume2)<1e-3  )
  return output 


def GetBoundingBoxProperties(Obj, key) : 
  bbox = vtk.vtkBoundingBox()
  bbox.SetBounds( Obj.GetBounds() )
  """ 
  Properties = {} 
  Properties['lengths'] = np.zeros(3)  
  Properties['center']  = np.zeros(3)  

  bbox.GetLengths( Properties['lengths'] )
  bbox.GetCenter( Properties['center'] ) #(center) 
  """ 
  
  array = np.zeros(3)  
  if key=='lengths': bbox.GetLengths( array )
  if key=='center' : bbox.GetCenter(  array )

  return array  


#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
def WithinExtents(_coords, _extents, _dim=3):
    outside = np.logical_or(_coords[:] < _extents[0::2], _coords[:] > _extents[1::2])
    inside  = np.all(~outside)
    return inside


def ShouldRefine(level, npoints, MAX_LEVEL, MAX_NPOINTS) : 
    ok = True
    if level   >= MAX_LEVEL   : ok = False
    if npoints <  MAX_NPOINTS : ok = False
    return ok


def HandleNode(cursor, points) :
    level    = cursor.GetLevel()
    idx      = cursor.GetGlobalNodeIndex()
    isLeaf   = cursor.IsLeaf()               # Is the cursor pointing to a leaf?
    isRoot   = cursor.IsRoot()               # Is the cursor at tree root? 
    children = cursor.GetNumberOfChildren()

    cellBounds = [0, 0, 0, 0, 0, 0]
    cursor.GetBounds(cellBounds)             #  bounding box = (xmin,xmax, ymin,ymax, zmin,zmax) 
    #print("level:%d, idx:%d, leaf:%d, children:%d, isroot:%d" % (level,idx,isLeaf,children,isRoot) )

    InSide = [ WithinExtents(p,cellBounds) for ip,p in enumerate(points) ]
    points = points[InSide,:]
    #print("level:%d, idx:%d," %(level,idx), points.shape ) 

    if isLeaf:
      if ShouldRefine(level, points.shape[0], MAX_NPOINTS=1, MAX_LEVEL=10):
        cursor.SubdivideLeaf()
        HandleNode(cursor, points)
    else:
      for childIdx in range(children):
        cursor.ToChild(childIdx)
        HandleNode(cursor, points)
        cursor.ToParent()


class GetHyperTreeGrid :   
  def __init__(self, coords) :
    htg = self.InitAsOctree()  
    self.FillTree(htg,coords)
    self.htg = htg  
    return


  def InitAsOctree(self): 
    ##  OCTREE, NO TOUCH!!  
    N   = 2              

    ## 1.0.  
    X   = np.linspace(0.0,1.0,N+1); 
    xValues = vtk.vtkDoubleArray()
    xValues.SetNumberOfValues( N+1 )
    for ix,x in enumerate(X): xValues.SetValue(ix,x) 

    ## 2.0.  
    htg = vtk.vtkHyperTreeGrid();     # [1]
    htg.Initialize()
    htg.SetDimensions([N+1,N+1,N+1])
    htg.SetBranchFactor(2)

    htg.SetXCoordinates(xValues)  
    htg.SetYCoordinates(xValues)  
    htg.SetZCoordinates(xValues)  
    print("\t[GetHyperTreeGrid] Extent:", htg.GetExtent() )
    print("\t[GetHyperTreeGrid] Bounds:", htg.GetBounds() )
    return htg  


  def FillTree(self, htg, coords):
    offsetIndex = 0  
    cursor      = vtk.vtkHyperTreeGridNonOrientedGeometryCursor()
    for treeId in range(htg.GetMaxNumberOfTrees()):
    ##print("treeId:", treeId )
      htg.InitializeNonOrientedGeometryCursor(cursor, treeId, True)
      cursor.SetGlobalIndexStart(offsetIndex)
      HandleNode( cursor, coords )
      offsetIndex += cursor.GetTree().GetNumberOfVertices()
    ##print("")
    #level    = cursor.GetLevel() # ??
    print('TotalNumberOfVertices:%d ' % offsetIndex)  
    return 


  def Print(self):
    htg = self.htg 
    print("Extent:", htg.GetExtent() )
    print("Bounds:", htg.GetBounds() )
    return 


  def Save(self, fname="hiperTree"):
    htg = self.htg
    Writer(htg, fname)
    return 


  def GetVtu(self) :
    vtu = vtk.vtkHyperTreeGridToUnstructuredGrid() 
    vtu.SetInputData( self.htg );
    vtu.Update();
    return vtu.GetOutputDataObject(0)  


  def SaveVtu(self, fname="hiperTree") :
    Writer(self.GetVtu(), fname)



#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
class GetPolyDataVertex :

  def __init__(self):
    self.Points    = None 
    self.Vertices  = None 
    self.Lines     = None 

    self.Mesh = vtk.vtkPolyData()
    #self.Mesh.SetLines(self.Lines)   
    return  


  def SetVertices(self, Pts, Ids=[]) :  
    self.Points   = vtk.vtkPoints()
    self.Vertices = vtk.vtkCellArray()
    self.PolyLine = vtk.vtkCellArray() 

    nIds = len(Ids)
    if nIds : assert(len(Pts)==len(Ids))  

    Order = {}  
    PolyLine = vtk.vtkPolyLine()
    PolyLine.GetPointIds().SetNumberOfIds( len(Pts) ); 
    for ipt,pt in enumerate(Pts) : 
      ## X.1. --- 
      idx  = self.Points.InsertNextPoint(pt)  
      if nIds : Order[Ids[ipt]] = idx

      ## X.1. --- 
      vert = vtk.vtkVertex()
      vert.GetPointIds().SetId(0,idx)  
      self.Vertices.InsertNextCell(vert) 

      ## X.1. --- 
      PolyLine.GetPointIds().SetId(idx,idx);  
    self.PolyLine.InsertNextCell(PolyLine); 

    self.Mesh.SetPoints(self.Points)   
    #self.Mesh.SetVerts(self.Vertices)   
    self.Mesh.SetLines(self.PolyLine)
    self.Mesh = GetArcLength(self.Mesh)

    self.Order = Order    
    return 


  def SetLines(self, Lines) :
    if(isinstance(Lines,list)) : Lines = np.array(Lines)
    assert( Lines.ndim==2 )
    assert( len(self.Order) )

    ## X.1. ---
    f     = lambda k : self.Order.get(k,-1)
    Lines = np.vectorize(f)(Lines); print( Lines )

    self.Lines = vtk.vtkCellArray() 
    for line in Lines : 
      l = vtk.vtkLine() 
      l.GetPointIds().SetId(0,line[0]);  
      l.GetPointIds().SetId(1,line[1]);    
      self.Lines.InsertNextCell(l)

    self.Mesh.SetLines(self.Lines)   
    return


  def AddCellArray(self, array, key) :
    if(isinstance(array,list)) : array = np.array(array)
    assert(self.Mesh.GetNumberOfCells()==array.shape[0])  

    if array.ndim==1: vtk_array = GetVtkArray(array, key)  
    else :            vtk_array = GetVtkArray(array.flatten(), key, ndim=array.shape[1])
    self.Mesh.GetCellData().AddArray(vtk_array)
    return


  def AddPointArray(self, array, key) :
    if(isinstance(array,list)) : array = np.array(array)
    assert(self.Mesh.GetNumberOfPoints()==array.shape[0])

    if array.ndim==1: vtk_array = GetVtkArray(array, key)
    else :            vtk_array = GetVtkArray(array.flatten(), key, ndim=array.shape[1])
    self.Mesh.GetPointData().AddArray(vtk_array)
    return


  def GetMesh(self) :
    return self.Mesh 


  def Save(self, fin) :  
    #self.Mesh = GetArcLength(self.Mesh)   
    Writer(self.Mesh, fin)  




#--------------------------------------------------------------------------||--#
## ToSEE :
  
## https://lorensen.github.io/VTKExamples/site/Cxx/Meshes/FillHoles/
vtk.vtkFillHolesFilter 

## https://lorensen.github.io/VTKExamples/site/Cxx/VisualizationAlgorithms/ClipSphereCylinder/
vtk.vtkClipPolyData 

## http://nealhughes.net/parallelcomp2/
## https://stackoverflow.com/questions/13068760/parallelise-python-loop-with-numpy-arrays-and-shared-memory
#from cython.parallel cimport prange

try : 
  import pyximport 
  from cython.parallel import prange
except:
  pass 

"""
cdef int i
cdef int n = 30
cdef int sum = 0
for i in prange(n, nogil=True): sum += i
print(sum)
"""


#--------------------------------------------------------------------------||--#
class Profiling : 
    def __init__(self) :
      import cProfile
      self.pr = cProfile.Profile()

      from io import BytesIO as StringIO
      self.s = StringIO()

      return 


    def Init(self) : 
      self.pr.enable()
      return


    def End(self):
      import pstats
      self.pr.disable() 

      #self.ps = pstats.Stats(self.pr, stream=self.s).sort_stats('tottime')
      self.ps = pstats.Stats(self.pr, stream=self.s).sort_stats("cumulative")
      self.ps.print_stats() 
# 
#      text = self.s.getvalue()  
#      text = text.split("\n") 
#      for t in text : print( t )  
# 
      """  
      import pandas as pd 
      df = pd.read_csv('output.txt', skiprows=5, sep='    ', names=['ncalls','tottime','percall','cumti    me','percall','filename:lineno(function)'])
      df[['percall.1', 'filename']] = df['percall.1'].str.split(' ', expand=True, n=1)
      df = df.drop('filename:lineno(function)', axis=1)
      print( df )
      """
 
      with open('test.txt', 'w+') as f:
        f.write( self.s.getvalue() )

      return






#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
"""
lldb /Users/poderozita/z2019_1/REPOSITORY/PV560_1/bin/pvpython PYs/pv_testBooleanSegmentationFault01_01.py 

NOTES: 
1. https://calcul.math.cnrs.fr/attachments/spip/IMG/pdf/vtk_visualization_pipeline.pdf
   https://blog.kitware.com/defining-time-varying-sources-with-paraviews-programmable-source/
  ts = outInfo.Get(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()  

2. Example for vtkXMLPStructuredGridWriter
  https://stackoverflow.com/questions/24123432/composing-vtk-file-from-multiple-mpi-outputs
  https://searchcode.com/codesearch/view/115849751/
  https://cmake.org/Wiki/VTK/Examples/Cxx/IO/XMLPImageDataWriter

3.  Distribute data among processors.   
  https://vtk.org/doc/release/5.2/html/a00336.html
  https://www.paraview.org/Wiki/VTK/Examples/Cxx/Meshes/OBBDicer 



SEE :
https://lorensen.github.io/VTKExamples/site/Cxx/Meshes/ClipDataSetWithPolyData/

"""
