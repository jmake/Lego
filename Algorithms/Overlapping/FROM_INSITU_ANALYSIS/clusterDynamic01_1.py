import numpy as np
import pandas as pd  
import itertools 
import os 
os.system("clear")

Array2tuples = lambda nparray : list(map(tuple,nparray))


class ClusterDynamicAnalysis : 
 
  def __init__(self, fin, minTimeStep, maxTimeStep) : 
    self.df                              = self.Load(fin, minTimeStep, maxTimeStep)   
    self.Sources,self.Targets,self.Nodes = self.GetNodes( self.df )
    self.Connections                     = self.GetConnections( self.df, self.Nodes) 
    self.Paths                           = self.GetPaths(self.Connections, self.Nodes)
    self.PathsWithBifurcations           = self.GetPathsWithBifurcations(self.Connections, self.Paths)  
    return  


  def Load(self, fin, minTimeStep, maxTimeStep):
    data = np.loadtxt(fin)
    df0  = pd.DataFrame(data) 
    df0  = df0[[0,1,2,3]].applymap(np.int64)
    df0.rename(columns={0:"timei", 1:"clusteri", 2:"timej", 3:"clusterj"},inplace=True)
    print("nLoad : %d" % len(df0) )

    df1   = df0.copy();
    if maxTimeStep : df1 = df1[ df1["timei"] <= maxTimeStep ]
    if minTimeStep : df1 = df1[ df1["timei"] >= minTimeStep ]
    #print(df1); #exit()

    return df1    


  def GetNodes(self, df0) : 
    """
    1. Sources and Targets :  
       a. Drop all duplicates    
       b. Columns must have different names 
       c. Index   must have      same names  
 
    2. Merge two dataframes by index :  
       a.  merge : inner join
       b.   join : left join 
       c. concat : outer join  

    3. No duplicates, double check 

    ## 6074
    df3 = pd.merge(Sources, Targets, left_index=True, right_index=True, how='inner')
    print("merge_inner:", len(df3))

    df4 = Sources.join(Targets, how='inner')
    print("join_inner:", len(df4) )

    df5 = pd.concat([Sources,Targets], axis=1, join='inner')
    print("concat_inner1:", len(df5) )

    ## 6704   
    df3 = pd.merge(Sources, Targets, left_index=True, right_index=True, how='outer')
    print("merge_outer:", len(df3) )
    
    df5 = pd.concat([Targets,Sources], axis=1, join='outer')
    print("concat_outer2:", len(df5) )

    df4 = Targets.join(Sources, how='outer')
    print("join_outer2:", len(df4) )

    """  
    targets = ["timei","clusteri"]
    sources = ["timej","clusterj"]
    keys    = ["time" ,"cluster" ]  

    ## X.1. Set index  
    df1     =  df0.copy(); #print( df1 )
    Sources = df1[sources].reset_index().set_index(sources)
    Targets = df1[targets].reset_index().set_index(targets)

    ## X.1. Droping duplicates  
    Sources = Sources[~Sources.index.duplicated(keep='first')]; #print( len(Sources) ) 
    Targets = Targets[~Targets.index.duplicated(keep='first')]; #print( len(Targets) )  

    ## X.1. Setting names (columns with different names, indices with same names)
    Sources = Sources.rename(columns={'index':0}) 
    Targets = Targets.rename(columns={'index':1})  
    Sources.index.names = keys 
    Targets.index.names = keys 

    ## X.1. Merging dataframes by index    
    df5 = pd.concat([Targets,Sources], axis=1, join='outer'); #print("concat_outer2:", len(df5) )

    ## X.1. Dropping duplicated index (double check)   
    df6 = df5.groupby(df5.index).first(); #print("groupby:", len(df6) ) 
    assert( len(df5) == len(df6)  )

    df7 = df5[~df5.index.duplicated()]; #print("duplicated:", len(df7) )
    assert( len(df5) == len(df7)  )

    ## X.1. Nodes dataframe  
    df7['nodes'] = np.arange(len(df7)); #print( df7 )  

    Group = df7['nodes'].groupby( df7.index )
    #for k,v in Group : print(k, v.values.tolist() ) 
    Nodes = {k:v.values.tolist() for k,v in Group} 
    Nodes = {k:{i:v for i,v in enumerate(V)} for k,V in Nodes.items()}
    Nodes = pd.DataFrame.from_dict(Nodes, orient='index') 
 
    Nodes = Nodes.rename(columns={0:'index'})
    Nodes.index.names = keys  
    print("nNodes : %d (Sources + Targets = %d + %d)" % (len(Nodes), len(Sources), len(Targets)) )

    Nodes.rename(columns={'index':'nodes'}).to_csv("nodes.csv")
    return Sources, Targets, Nodes  



  def GetConnections(self, df1, Nodes) : 
    targets = ["timei","clusteri"]
    sources = ["timej","clusterj"]
    keys    = ["time" ,"cluster" ]

    ## X.1. 
    Dic1  = {}  
    Dic2  = {}
    Group = df1.groupby(sources)
    for k,V in Group :
      V      = V[targets]  
      V      = Array2tuples(V.values)  
      nodei  = Nodes.loc[k,:].values.flatten()[0]
      Dic1[nodei] = {i:Nodes.loc[v,:].values.flatten() for i,v in enumerate(V)}
      Dic2[k]     = {i:v for i,v in enumerate(V) }

    df1 = pd.DataFrame.from_dict( Dic1 ).T 
    df1 = df1.applymap(lambda x : np.nan if np.isnan(x) else x[0] ) 
    return df1  


  def GetPaths(self, Connections, Nodes) :
    import networkx as nx  

    Labels = Nodes['index'].values.tolist(); #print(Labels)
    Paths  = pd.DataFrame(index=Labels); #print( Paths ) 

    Connections = Connections.unstack().dropna()  
    Connections = Connections.reset_index() 
    Connections = Connections[ ['level_1',0] ]  
    Edges = Array2tuples(Connections.values); #print( Edges )   

    G      = nx.DiGraph( Edges )
    roots  = (v for v, d in G.in_degree() if d == 0)
    leaves = (v for v, d in G.out_degree() if d == 0)

    from functools import partial
    all_paths = partial(nx.all_simple_paths, G)

    from itertools import chain, product, starmap
    chaini = chain.from_iterable
    all_paths = list(chaini(starmap(all_paths, product(roots,leaves))))

    for ii,path in enumerate(all_paths) :
      npath = len(path)
      path  = np.array(path).astype(int); print("path(%03d):" % ii, path )
      Paths.loc[   :,ii] = False 
      Paths.loc[path,ii] = True    

    Paths.index.names = ['nodes'] 
    print("nPaths : %d " % len(Paths) )

    Paths.to_csv("paths.csv")   
    return Paths 


  def GetPathsWithNode(self, Paths, node, show=False) :
    #Paths   = self.Paths; #print( Paths ) 
    pathids = Paths.loc[node,:]
    pathids = pathids[pathids].index.values  
    nodes   = Paths.index.values

    PathsWithNode = {} 
    for pathid in pathids :  
      row  = Paths[pathid]   
      path = nodes[row]  
      PathsWithNode[pathid] = path 
      if(show) : print("\tnode:%d, Path(%d):" % (node,pathid), path )
    return PathsWithNode  


  def GetPathsWithBifurcations(self, Connections, Paths, show=False) :
    nConnections =  Connections.count(axis=1) ; #print( nConnections )
    nConnections = nConnections[ nConnections > 1 ]; #print( nConnections )  

    PathsWithBifurcations = {} 
    for node,nconnections in nConnections.iteritems() : 
      PathsWithNode = self.GetPathsWithNode(Paths, node)  
      print("node:%d, size:%d; npaths:%d " % (node,nconnections,len(PathsWithNode)) )

      for pathid,path in PathsWithNode.items():
        if(pathid in PathsWithBifurcations) : 
          ok = np.all(PathsWithBifurcations[pathid] == path)
          if(not ok) : print("\tpathid:%d path:" % pathid, ok ) 
          assert(ok)  
        else : 
          PathsWithBifurcations[pathid] = path 
    
      if(nconnections != len(PathsWithNode)) :
        for pathid,path in  PathsWithNode.items():  
          print("\tpathid:%d path:" % pathid, path )
    return PathsWithBifurcations  


class PlotyClusterDynamicAnalysis(ClusterDynamicAnalysis) :  

  def __init__(self, fin, minTimeStep=None, maxTimeStep=None) :  
    ClusterDynamicAnalysis.__init__(self, fin, minTimeStep, maxTimeStep) 
    self.PlotPaths( self.PathsWithBifurcations, self.Nodes)
    return  


  def PlotPaths(self, Paths, Nodes) :
    assert( isinstance(Paths,dict) )

    Targets   = []
    Sources   = []
    LinkNames = []
    for pathid,path in Paths.items() :
       sources = path[  :-1]
       targets = path[1:   ]
       for s in sources : Sources.append(s)
       for t in targets : Targets.append(t)
       for t in targets : LinkNames.append(pathid)

    Labels  = Nodes['index'].values.tolist()
    self.PlotSankey(Sources, Targets, Labels, LinkNames)
    return


  def PlotSankey(self, Sources, Targets, Labels, LinkNames=None) :
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ## X.0.
    Values = list(range( len(Targets) ))

    print(" Labels:", len(Labels) )
    print("Sources:", len(Sources) )
    print("Targets:", len(Targets) )

    ## X.0.
    sankey = self.GetSankey(Labels, Sources, Targets, Values, LinkNames)
    table  = self.GetTable( self.df )

    hovertemplate  = "Link from node %{source}"
    hovertemplate += "to node%{target} has value %{label}"

    fig = go.Figure() #sankey) 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, specs=[[{"type":"sankey"}],[{"type":"table"}]],)
    fig.add_trace(sankey, row=1, col=1)
    fig.add_trace( table, row=2, col=1)
    fig.update_layout(title_text="Sankey Diagram", font_size=10, hovermode = 'closest',
                      xaxis=dict(rangeslider=dict(visible=True,autorange=True,))
                     )
    fig.write_html("clusterDynamic01_1.html")
    fig.show()
    ## 
    return


  def GetSankey(self, Labels, Sources, Targets, Values, LinkNames) :
    import plotly.graph_objects as go

    ## X.0.
    link   = dict(source=Sources, target=Targets, value=Values, label=LinkNames) #, hovertemplate=hovertemplate)
    node   = dict(label=Labels, pad=15, thickness=20, color="blue", line=dict(color="black",width=0.5))
    sankey = go.Sankey(link=link, node=node)
    return sankey   


  def GetTable(self, df): 
    import plotly.graph_objects as go
 
    ## X.0.
    header = dict(values=list(df.columns ) ); #print( header ) 
    cells  = dict(values=list(df.T.values) )  
    table  = go.Table(header=header, cells=cells)  

    if 0 : 
      fig = go.Figure(table) 
      fig.update_layout(width=500, height=300)
      fig.show()
      exit() 
 
    return table

 

## X.1. 
fin  = "voxels_clusterDynamic.dat"  
pcda = PlotyClusterDynamicAnalysis(fin, minTimeStep=None, maxTimeStep=None)




 
