import numpy as np 
import pandas as pd 
import glob 
import re 
import os; os.system("clear")
import vtk_tools


#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
class ClusterExtractPaths : 

  def __init__(self) :   
    return


  def Loading(self) : 
    self.Vtms  = self.LoadingVtmClusters()  
    self.Nodes = self.LoadingCsvNodes()  
    self.Paths = self.LoadingCsvPaths()                     ## Paths = [node1,node2,...,nodeN] 
    self.Paths,self.Nodes = self.GetTimeCluster(self.Paths,self.Nodes) ## Paths = [(time1,cluster1),time2,cluster2),...,timeN,clusterN)] 
    return

 
  def LoadingVtmClusters(self) :
    fin   = "*_clusters.vtm"  
    Files = glob.glob(fin)   
    Files = {int(re.findall('\d+',f)[-1]):f for f in Files}; 
    Vtms  = {k:vtk_tools.Reader(v) for k,v in Files.items()}  
    Vtms  = {k:vtk_tools.GetBlocks(v) for k,v in Vtms.items()}  
    return Vtms 


  def LoadingCsvNodes(self) : 
    fin   = "nodes.csv"
    Nodes = pd.read_csv(fin, index_col='nodes');  
    return Nodes  


  def LoadingCsvPaths(self) :  
    fin   = "paths.csv" 
    Paths = pd.read_csv(fin, index_col='nodes') 
    nodes = Paths.index.values  
    Paths = {pathid:nodes[path.values] for pathid,path in Paths.items()}    
    return Paths 


  def GetTimeCluster(self, Paths, Nodes): 
    assert( isinstance(Paths,dict) ) 
    assert( isinstance(Nodes,pd.DataFrame) )

    NewNodes = {} 
    NewPaths = {} 
    for pathid,path in Paths.items() :
      NewNodes[pathid] = path 
      path             = Nodes.loc[path,:].values; 
      NewPaths[pathid] = vtk_tools.Array2tuples(path) 
    return NewPaths,NewNodes 


  def Walking(self, show=False) : 
    Walks = {}
    for pathid,path in self.Paths.items() :
      if show : print("pathid:", pathid, end=" ")

      walk = []
      for node in path :
        itime    = node[0]
        icluster = node[1]
        if show : print("(%d,%d) -" % (itime,icluster), end=" ")

        current = self.Vtms.get(itime,None); assert(current)
        cluster = current.get(icluster,None); assert(cluster)
        walk.append(cluster)
      Walks[pathid] = walk
      if show : print()  

    self.Walks = Walks 
    return self.Walks   


  def SaveClustersPaths(self, fname) : 
    assert(isinstance(self.Walks,dict))
    for k,V in self.Walks.items() :
      Vtm = vtk_tools.GetMultiBlockDataSetFromList(V)   
      vtk_tools.Writer(Vtm,fname+"%03d" % int(k))  
    return 


  def SaveCenterOfMassPaths(self, fname) :
    assert(isinstance(self.Walks,dict))

    #pathid = '254' 
    #vtp = self.SaveCenterOfMassPath(pathid) #, fname+"%03d"%int(pathid) ) 

    #for pathid in self.Paths.keys() :
    #  vtp = self.SaveCenterOfMassPath(pathid, fname+"%03d"%int(pathid) ) 

    Vtps = {int(pathid):self.SaveCenterOfMassPath(pathid) for pathid in self.Paths.keys()} 
    Vtm  = vtk_tools.GetMultiBlockDataSet(Vtps)   
    vtk_tools.Writer(Vtm,fname)   
 
    return


  def SaveCenterOfMassPath(self, pathid, fname=None, show=False) : 
      Path   = self.Paths.get(pathid,None); #assert(Path)   
      Nodes  = self.Nodes.get(pathid,None); #assert(Nodes)   
      assert(len(Path)==len(Nodes))

      ## X.1. --- 
      CenterOfMass = []     
      Lengths      = []     
      Times        = []   
      Ids          = []  
      for path,node in zip(Path,Nodes) :
        itime    = path[0]; Times.append(itime)    
        icluster = path[1]; Ids.append(icluster)  
        if show : print("%d:(%d,%d) " % (node,itime,icluster), end=" ")

        current = self.Vtms.get(itime,None); assert(current)
        cluster = current.get(icluster,None); assert(cluster)  
        cluster = vtk_tools.SetFieldData(cluster,np.array([itime]),"TimeValue")  
        com     = vtk_tools.GetCenterOfMassFilter(cluster); CenterOfMass.append(com) 
        lengths = vtk_tools.GetBoundingBoxProperties(cluster,"lengths"); Lengths.append(lengths) 
        if show : print(com, lengths) 

      ## X.1. --- 
      CenterOfMass = np.array(CenterOfMass); assert(CenterOfMass.ndim==2) 
      Ids          = np.array(Ids); assert(Ids.ndim==1)  
      Lines        = np.stack( (Ids[:-1],Ids[1:]), axis=1) 

      ## X.1. --- 
      Poly = vtk_tools.GetPolyDataVertex() 
      Poly.SetVertices( CenterOfMass, Ids )    
      Poly.AddCellArray( [int(pathid)], "PathId")
      Poly.AddPointArray( Lengths, "Lengths" )   
      Poly.AddPointArray( Times, "Times" )
      Poly.AddPointArray( Nodes, "Nodes" )
      Poly.AddPointArray( Ids, "Ids" )
      if fname : Poly.Save(fname)  
 
      return Poly.GetMesh() 




#--------------------------------------------------------------------------||--#
extracting = ClusterExtractPaths()  
extracting.Loading()   
extracting.Walking()   
#extracting.SaveClustersPaths("clustersPaths") 
extracting.SaveCenterOfMassPaths("centerofmassPaths")  


## X.1. ---  
## X.1. --- 
#--------------------------------------------------------------------------||--#
#--------------------------------------------------------------------------||--#
"""

time ~/z2019_1/REPOSITORY/PV560_1/EXECS01/paraview.app/Contents/bin/pvpython ../clusterExtractPaths01_1.py  

time /Applications/ParaView-5.7.0-RC1.app/Contents/bin/pvpython ../clusterExtractPaths01_1.py 

time python ../clusterExtractPaths01_1.py 

"""
