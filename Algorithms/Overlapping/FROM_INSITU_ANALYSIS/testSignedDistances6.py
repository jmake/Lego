import numpy as np 
import os 
import glob
import vtk 
import vtk_tools
from vtk.util import numpy_support
import cProfile
import re

#--------------------------------------------------------------------------||--#
""" 
## ParaView-5.6.0 : 
##   AttributeError: 'module' object has no attribute 'vtkCollisionDetectionFilter'

import sys 

XYZ_PATH="WHERE/IS/VtkCollisionDetectionFilter/??"
sys.path.append(XYZ_PATH) #os.path.join(os.getcwd(), './'))

import vtkXYZPython; #help(vtkXYZPython.vtkCollisionDetectionFilter) 

def CollisionDetection(vtpA, vtpB, tap='\t') :
    #Writer(vtpA, "a")
    #Writer(vtpB, "b")

    matrix1 = vtk.vtkMatrix4x4()
    transform0 = vtk.vtkTransform()

    collide = vtkXYZPython.vtkCollisionDetectionFilter()
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

""" 
#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
class ExtractClusters : 

  def __init__(self, fin, prop, threshold):   
    #self.source    = self.Loading(fin) 
    self.fin       = fin  
    self.property  = prop
    self.threshold = threshold  

    self.Clusters      = None   
    self.CenterOfMassI = None 

    self.ClustersIJ       = {}  
    self.ClustersIJmaxLen = 0      
    return 


  def Loading(self) :
    self.Identifiers = "vtkIdFilter_Ids" #"Identifiers" 
    self.Source      = vtk_tools.Reader(self.fin)
    self.Source      = vtk_tools.SetIdentifiers(self.Source, key=self.Identifiers) #"Identifiers") 
    return self.Source


  def GetVoxelization(self, x, y, z, n, fin=None) : 
    Grid = vtk_tools.GetResampleToImage(self.Source, n, self.Source.GetBounds() )
    Grid = vtk_tools.SetIdentifiers(Grid, key=self.Identifiers) #"Identifiers") 
    print("\t[GetVoxelization] Extent:", Grid.GetExtent() )
    print("\t[GetVoxelization] Bounds:", Grid.GetBounds() )    
    if(fin): vtk_tools.Writer(Grid, fin)
    return Grid 



  def GetClusters(self, isosurface, key, fin=None) :
    array        = vtk_tools.GetCellDataArray(isosurface,key)
    arrayDic     = vtk_tools.getDicFromArray(array)
    Clusters     = {k:vtk_tools.ExtractCluster(isosurface,v,"CELL") for k,v in arrayDic.items()}

    if(fin) :
      ClustersVtm = vtk_tools.GetMultiBlockDataSet(Clusters)
      vtk_tools.Writer(ClustersVtm,  fin+"_clusters")
    return Clusters


  def GetOutline(self, clusters=None, fin=None) :
    assert(isinstance(clusters,dict)) 

    Ids      = [k for k,v in clusters.items()]
    Outlines = [vtk_tools.GetOutline(clusters[Id]) for Id in Ids]

    if(fin) :
      OutlinesVtm = vtk_tools.GetMultiBlockDataSetFromList(Outlines)
      vtk_tools.Writer(OutlinesVtm,  fin+"_outlines")
    return Outlines  


  def GetIsoSurface(self, source=None, fin=None) :  
    source     = source if source else self.Source    
    isoSurface = vtk_tools.GetCountour(source, self.property, self.threshold) 
    isoSurface = vtk_tools.GetConnectivity(isoSurface)
    isoSurface = vtk_tools.Vtx2Vtp(isoSurface)
    if(fin): vtk_tools.Writer(isoSurface, fin+"_isosurface")
    return isoSurface  


  def GetThreshold(self, source=None, fin=None) : 
    source    = source if source else self.Source
    threshold = vtk_tools.GetThreshold(source, key=self.property, lower=None, upper=self.threshold)
    #threshold = vtk_tools.Vtx2Vtp(threshold)
    #threshold = vtk_tools.CleanPolyData(threshold)
    threshold = vtk_tools.GetConnectivity(threshold)
    if(fin): vtk_tools.Writer(threshold, fin+"_threshold")
    return threshold  


  def SignedDistance(self, isosurface, source=None, fin=None) :  
    source    = source if source else self.Source
    sdistance = vtk_tools.GetSignedDistances(isosurface, source)
    source.GetPointData().SetScalars(sdistance)  

    inSide,outSide = vtk_tools.GetClipper(source,0.0)
    inSide = vtk_tools.GetConnectivity(inSide)  
    if(fin): vtk_tools.Writer(inSide, fin+"_signed")  
    return inSide    


  def GetCenterOfMass(self, fin=None) :   
    clusters = self.Clusters
    assert(isinstance(clusters,dict))
    CenterOfMassI  = {k:vtk_tools.GetCenterOfMassFilter(v)              for k,v in clusters.items()}

    if fin :
      ## X.1  
      Points   = vtk.vtkPoints()
      Vertices = vtk.vtkCellArray()

      for ii1,clusteri in enumerate(sorted(clusters)) :
        ## X.1  
        ii2 = Points.InsertNextPoint( CenterOfMassI[clusteri] )
        assert(ii1==clusteri)
        assert(ii2==clusteri)

        ## X.1  
        vert = vtk.vtkVertex()  
        vert.GetPointIds().SetId(0,clusteri)
        Vertices.InsertNextCell(vert)  

      ## X.1  
      CoM = vtk.vtkPolyData()
      CoM.SetPoints(Points)  
      CoM.SetVerts(Vertices)  
      #CoM.GetCellData().AddArray( Lengths )
      #CoM.GetCellData().AddArray( ClustersJ )
      vtk_tools.Writer(CoM, fin+"_centerofmass")

    if not self.CenterOfMassI : self.CenterOfMassI = CenterOfMassI
    return CenterOfMassI


  def GetCenterOfMassProperties(self, CenterOfMassJ=None, fin=None) :
    clusters = self.Clusters
    assert(isinstance(clusters,dict))

    ## X.1  
    PropertiesI         = {k:vtk_tools.GetBoundingBoxProperties(v,'lengths') for k,v in clusters.items()}
    CenterOfMassI       = self.GetCenterOfMass(fin) 
    self.CenterOfMassIJ = self.ConnectCenterOfMass(CenterOfMassI,CenterOfMassJ) if CenterOfMassJ else {}
 
    ## X.1  
    if fin and CenterOfMassJ : 
      ## X.1  
      Points    = vtk.vtkPoints()
      Vertices  = vtk.vtkCellArray()
      Lines     = vtk.vtkCellArray()

      for clusteri in sorted(CenterOfMassI) : Points.InsertNextPoint( CenterOfMassI[clusteri] ) 
      VtkClustersJ = {clusterj:Points.InsertNextPoint( CenterOfMassJ[clusterj] ) for clusterj in sorted(CenterOfMassJ) }

      ## X.1   
      ClustersIJ = self.ClustersIJ.copy()
      ClustersIJ = {clusteri:[VtkClustersJ.get(clusterj,None) for clusterj in clustersj] for clusteri,clustersj in ClustersIJ.items()}
      for clusteri,clustersj in ClustersIJ.items() :
        for clusterj in clustersj : 
          #print( clusteri, clusterj )
          line = vtk.vtkLine()  
          line.GetPointIds().SetId(0,clusteri);  
          line.GetPointIds().SetId(1,clusterj);   
          Lines.InsertNextCell(line)   
 
      ## X.1   
      CoM = vtk.vtkPolyData()
      CoM.SetPoints(Points)
      CoM.SetVerts(Vertices)
      CoM.SetLines(Lines)  

      ## X.1  
      vtk_tools.Writer(CoM, fin+"_connectivities")  

    if not self.CenterOfMassI : self.CenterOfMassI = CenterOfMassI
    return CenterOfMassI


  def ConnectCenterOfMass(self, CenterOfMassI, CenterOfMassJ) : 
    assert( isinstance(self.ClustersIJ   ,dict) ) ## From 'GetOverlapping'  
    ClustersIJ = self.ClustersIJ   
 
    ComIJ = {} 
    for clusteri,clustersj in ClustersIJ.items() :
      comij = {}  
      for clusterj in clustersj : 
        comi = CenterOfMassI.get(clusteri,None); assert(isinstance(comi,np.ndarray))  
        comj = CenterOfMassJ.get(clusterj,None); assert(isinstance(comj,np.ndarray)) 
        comij[clusterj] = comj - comi
      ComIJ[clusteri] = comij   
    return ComIJ 


#--------------------------------------------------------------------------||--#
class ExtractClustersByThreshold(ExtractClusters) :

  def __init__(self, fin, prop, threshold) :  
    ExtractClusters. __init__(self, fin, prop, threshold)  
    return


  def GetClusteringAnalysis(self, fin=None) :
    self.Threshold     = self.GetThreshold(self.Source) #, fin)
    self.Clusters      = self.GetClusters(self.Threshold, "RegionId", fin) 
    self.ClusterIdsByCells,self.CellIdsByCluster = self.GetClusterIdsByCells(self.Clusters)
    return


  def GetClusterIdsByCells(self, clusters):
    assert(isinstance(clusters,dict))
    """
    'Identifiers' contains the 'original' cell 'identifiers' (numeration) so that
    'CellIdsByCluster'  will allocate    'cell ids' contained inside a cluster given 'clusterid'.  
    'ClusterIdsByCells' will allocate 'cluster ids' contained inside a    cell given    'cellid'.  
    """
    CellIdsByCluster  = {}
    ClusterIdsByCells = {}
    for clusterid,cluster in clusters.items() :
      cellIdsByCluster = vtk_tools.GetCellDataArray(cluster,self.Identifiers) #"Identifiers")
      cellIdsByCluster = np.sort(cellIdsByCluster)
      unique = np.unique(cellIdsByCluster)
      assert( unique.shape[0]==cellIdsByCluster.shape[0] )  ## ??  

      for icell,cellid in enumerate(cellIdsByCluster) :
        if ClusterIdsByCells.get(cellid,None) : ClusterIdsByCells[cellid].append(clusterid)
        else : ClusterIdsByCells[cellid] = clusterid

      CellIdsByCluster[clusterid] = cellIdsByCluster
    return ClusterIdsByCells, CellIdsByCluster


  def SetOverlapping(self):
    return self.ClusterIdsByCells    


  def GetOverlapping(self, PreviousClusterIdsByCells) :
    """
    'PreviousClusterIdsByCells' : dict containing 'clusters ids' inside a given cell  
    'cellIdsByCluster'          : list of cells contained in cell 'clusterid' 

    For each cell 'cellid' contained in a given cluster 'clusterid' 
    check 'previous clusters' allocated inside 'cellid'  

    ClustersJIntoClusterI = {}  
    for cellid in CellIdsByCluster[clusterid] : 
      previousClusters = PreviousClusterIdsByCells[cellid]
      ClustersJIntoClusterI[clusterid] = previousClusters   
       

    'ClustersJIntoClusterI' contains 'previous clusters' clustersJIntoClusterI  
                            which overlaps the 'local cluster' clusterid  

    'ClustersJ' contains 'previous clusters' ids 
                which overlaps the 'local cluster' clusterid  


    """
    ## X.0. CellIdsByCluster -> ClustersJIntoClusterI  
    ClustersJIntoClusterI = {} 
    for clusterid, cellIdsByCluster in self.CellIdsByCluster.items():
      clustersJIntoClusterI = [PreviousClusterIdsByCells.get(cellid,np.nan) for cellid in cellIdsByCluster]
      clustersJIntoClusterI = np.unique(clustersJIntoClusterI)
      clustersJIntoClusterI = clustersJIntoClusterI[ np.isfinite(clustersJIntoClusterI) ]
      ClustersJIntoClusterI[clusterid] = clustersJIntoClusterI.astype(int)

    self.ClustersIJ       = ClustersJIntoClusterI
    self.ClustersIJmaxLen = np.amax([len(clustersj)  for clusteri,clustersj in self.ClustersIJ.items()])  
    return self.ClustersIJ 


#--------------------------------------------------------------------------||--#
class ExtractClustersByIsoSurface(ExtractClusters) :

  def __init__(self, fin, prop, threshold) :
    ExtractClusters. __init__(self, fin, prop, threshold)
    return


  def GetClusteringAnalysis(self, fin=None) :
    self.Threshold     = self.GetIsoSurface(self.Source, fin)
    self.Clusters      = self.GetClusters(self.Threshold, "RegionId", fin) #+"_threshold") 
    return


  def GetClusterIdsByCells(self, clusters):
    CellIdsByCluster  = {}
    ClusterIdsByCells = {}  
    return ClusterIdsByCells, CellIdsByCluster


  def SetOverlapping(self) : 
    return self.Clusters #self.ClusterIdsByCells


  def GetOverlapping(self, PreviousClusters) : 
    """  
    'ClustersJ' contains 'previous clusters' ids 
                which overlaps the 'local cluster' clusterid  
    """
    assert(isinstance(PreviousClusters,dict))
    assert(isinstance(self.Clusters   ,dict))
    ClustersJ = PreviousClusters 
    ClustersI = self.Clusters 

    self.ClustersIJ       = self.OverlappingWithBBoxes(ClustersI, ClustersJ)
    self.ClustersIJ       = self.OverlappingWithClusters(ClustersI, ClustersJ, self.ClustersIJ)  
    self.ClustersIJmaxLen = np.amax([len(clustersj)  for clusteri,clustersj in self.ClustersIJ.items()])  
    return self.ClustersIJ


  def OverlappingWithBBoxes(self, ClustersI, ClustersJ) : 
    ClustersIJ = {}  

    for Idi,Clusteri in ClustersI.items() :  
      bboxi = Clusteri.GetBounds(); # [xmin, xmax, ymin, ymax, zmin, zmax]
      for Idj,Clusterj in ClustersJ.items() :
        bboxj = Clusterj.GetBounds(); # [xmin, xmax, ymin, ymax, zmin, zmax]
        overlapping_x = self.BBoxesTest(bboxi[0:2],bboxj[0:2])  
        overlapping_y = self.BBoxesTest(bboxi[2:4],bboxj[2:4])  
        overlapping_z = self.BBoxesTest(bboxi[4:6],bboxj[4:6])  
       #overlapping   = overlapping_x or  overlapping_y or  overlapping_z; ## ??  
        overlapping   = overlapping_x and overlapping_y and overlapping_z; ## ??   
        if(overlapping) :
          if ClustersIJ.get(Idi,None) : ClustersIJ[Idi].append(Idj) 
          else : ClustersIJ[Idi] = [Idj]  
          #print(  Idi, bboxi  )
          #print(  Idj, bboxj  )
    return ClustersIJ  
 

  def OverlappingWithClusters(self, ClustersI, ClustersJ, ClustersIJ) :
    """ 
    + SEE: 
        /afs/pdc.kth.se/home/m/miguelza/z2019_3/REPOSITORY/PAAKAT/2019AUG09/TESTS/PV560/FORTRAN/E01_BASE/parallel_clusteranalysis.hpp 
        /Users/poderozita/Dropbox/PythonScrits/z2019_2/pv_testSplitCluster01_01.py 
 
    + vtkExtractEnclosedPoints 
        https://lorensen.github.io/VTKExamples/site/Python/PolyData/CellsInsideObject/  

    """

    ClustersIJ_New = {} 
    for Idi,Idsj in ClustersIJ.items() :
      Clusteri = ClustersI.get(Idi,None); assert(Clusteri) 
      Clusteri = vtk_tools.GetTriangles(Clusteri)
      for Idj in Idsj :  
        Clusterj = ClustersJ.get(Idj,None); assert(Clusterj)
        Clusterj = vtk_tools.GetTriangles(Clusterj)   
        Clusterk = CollisionDetection(Clusteri,Clusterj,"\t  |_") ## [vtkExtractEnclosedPoints] 
       #Clusterk = vtk_tools.CollisionDetection(Clusteri,Clusterj,"\t  |_") ## [vtkExtractEnclosedPoints] 
        if Clusterk :
          if ClustersIJ_New.get(Idi,None) : ClustersIJ_New[Idi].append(Idj) 
          else : ClustersIJ_New[Idi] = [Idj]  
    return ClustersIJ_New
 

  def BBoxesTest(self, a, b) :
    """  
    1. (amin <= bmax)&&(amax >= bmin)

           bmin     bmax 
            |--------|

        |--------|
      amin     amax 
  
    """  
    amin = a[0]; amax = a[1]; assert(amin < amax)   
    bmin = b[0]; bmax = b[1]; assert(bmin < bmax)    

    #print( "a:", a, "b:", b )
    overlapping = (amin <= bmax) and (amax >= bmin);
    return overlapping;




#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
## X.1. 
thre   = 0.1  
prop   = "VELOX" 
Extent = np.array([2, 2, 4]) * 25 * 2 # * 1 = 465; *2 = 

Voxelization = False  
#AnalyzeClusters =  ExtractClustersByThreshold; Voxelization = True    
AnalyzeClusters = ExtractClustersByIsoSurface
##AnalyzeClusters = ExtractClustersBySignedDistance 

 
## X.1. 
if 1 :
  os.system("rm -rf voxels*")

  profiling = vtk_tools.Profiling()
  profiling.Init()

  #Files = glob.glob("Voxels0128_1/*vti"); #print( Files ) 
  Files = glob.glob("0128/mesh*pvtu"); #print( Files )  

  Files = {int(re.findall('\d+',f)[-1]):f for f in Files}; #print( Files )
  Times = np.sort( list(Files.keys()) ); #print( Times )
  Times = Times[:2]
  Times = {current:previous for current,previous in zip(Times[1:],Times[0:])}; #print( Times )
  for current in sorted(Times) : print("previous -> current : %d -> %d" %(Times[current],current) )
 
  #exit()
  ConnectClustersIJ     = {}  
  ConnectCenterOfMassIJ = {}
  ConnectCenterOfMassI  = {}
  ConnectCenterOfMassJ  = {}
  for current in sorted(Times) : 
    previous = Times[current] 
    print("(current,previous) : (%d,%d)" % (current,previous) )

    ## X.1. Previous 
    PreviousFile = Files[previous]
    Previous     = AnalyzeClusters(PreviousFile, prop, thre)
    Previous.Loading()
    if Voxelization : Previous.Source = Previous.GetVoxelization([-1.0,1.0], [-1.0,1.0], [0.0,12.5], Extent, fin="voxelsPrevious%02d"%previous)
    Previous.GetClusteringAnalysis(fin="voxelsPrevious%02d"%previous)
 
    ## X.1. Current 
    CurrentFile  = Files[current]
    Current      = AnalyzeClusters(CurrentFile, prop, thre)
    Current.Loading()  
    if Voxelization : Current.Source = Current.GetVoxelization([-1.0,1.0], [-1.0,1.0], [0.0,12.5], Extent, fin="voxelsCurrent%02d"%current)
    Current.GetClusteringAnalysis(fin="voxelsCurrent%02d"%current) 
 
    ## X.1. Overlapping   
    ConnectClustersIJ[current] = Current.GetOverlapping( Previous.SetOverlapping() ) 

    ## CoMs  
    Previous.GetCenterOfMass(fin="voxelsPrevious%02d"%current)
    Current.GetCenterOfMassProperties(Previous.GetCenterOfMass(), fin="voxelsCurrent%02d"%current)    

    ConnectCenterOfMassJ[current]  = Previous.CenterOfMassI  
    ConnectCenterOfMassI[current]  =  Current.CenterOfMassI  
    ConnectCenterOfMassIJ[current] =  Current.CenterOfMassIJ 
 
  #exit() 
  ## X.1.  
  ClusterDynamic = []  
  for timei,ClustersIJ in ConnectClustersIJ.items() :   
    timej          = Times[timei]   
    CenterOfMassIJ = ConnectCenterOfMassIJ.get(timei,None); assert(CenterOfMassIJ) 
    CenterOfMassI  = ConnectCenterOfMassI.get(timei,None) 
    CenterOfMassJ  = ConnectCenterOfMassJ.get(timei,None)
    for i,(clusteri,clustersj) in enumerate(ClustersIJ.items()) : 
      for j,clusterj in enumerate(clustersj) :
        comi  = CenterOfMassI[clusteri]
        comj  = CenterOfMassJ[clusterj]
        comij = CenterOfMassIJ[clusteri][clusterj]
        assert( np.all(np.abs(comi+comij-comj)<=1e-3) ) 

        clusterDynamic = [timei,clusteri, timej,clusterj] + comi.tolist() + comj.tolist() 
        ClusterDynamic.append( clusterDynamic )
        print("(%d,%d) <- (%d,%d) : " % (timei,clusteri, timej,clusterj), comij, comi)

  np.savetxt("voxels_clusterDynamic.dat", ClusterDynamic) 
  #profiling.End()



#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
"""
+ running

  ## :)  
  time /Applications/ParaView-5.7.0-RC1.app/Contents/bin/pvpython testSignedDistances6.py 

  ## :(. AttributeError: 'module' object has no attribute 'vtkCollisionDetectionFilter'
  time /Applications/ParaView-5.6.0.app/Contents/bin/pvpython testSignedDistances6.py 



+ profiling :
  https://wiki.python.org/moin/PythonSpeed/PerformanceTips#Profiling_Code

+ CASE :
  /Users/poderozita/z2020_1/RUNNER/MARCO/DUCT01_1/SignedDistance06_1


+ FROM :
   /Users/poderozita/Dropbox/PythonScrits/z2020_1/TINO/

+ TESTED : 
  /Users/poderozita/z2020_1/RUNNER/MARCO/DUCT01_1/SignedDistance06_1


""" 
