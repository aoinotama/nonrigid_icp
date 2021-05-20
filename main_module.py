import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sksparse.cholmod import cholesky_AAt
import open3d as o3d
import copy

# configuration 
Debug = False
DebugByStep = True
DebugShowStep = False
DebugExportStep = True
normalWeighting = False
gamma = 1 # Weight Differences in the rotational and Skew part of the deformation against the translational part of the deformation
alphas_config = np.linspace(2000000000,1,20)
iterative_times_config = 3
folder = "Expression/Test/"
_Nose_Index = 114

Threhold_Nose = 0.2
Threhold_VertDis = 0.2


# solve MX - b = 0
def choleskySolveM(M, b):
    factor = cholesky_AAt(M.T)
    return factor(M.T.dot(b)).toarray()

# Reture the Matrix For Edge Connnection
# in-Edge == 1, ex-Edge == -1
# 
def EdgeMatrix(edges,vertices):
    n_source_edges = len(edges)
    n_source_vertices = vertices.shape[0]
    M = sparse.lil_matrix((n_source_edges,n_source_vertices),dtype=np.float32)
    for i,t in enumerate(edges):
        M[i,t[0]] = -1
        M[i,t[1]] = 1
        
    return M

def DisRelationMatrix(edges,vertices):
    n_source_edges = len(edges)
    n_source_vertices = vertices.shape[0]
    # Compute the distance To Nose
    VectorToNose = vertices - vertices[_Nose_Index]
    DisToNose = np.linalg.norm(VectorToNose,axis=1)
    MaxDis = max(DisToNose)
    # Set threhold according to MaxDis
    Relations = []
    for i in range(0,len(vertices)):
        for j in range(i+1,len(vertices)):
            if (DisToNose[i] < 0.8 * MaxDis):
                WeightNose = 1 - DisToNose[i]/MaxDis
                VertDis = np.linalg.norm(vertices[i]-vertices[j])
                WeightDis = 1 - VertDis/MaxDis
                if WeightDis < 0:
                    WeightDis = 0
                # Discard small Weight to Reduce 
                if WeightNose * WeightDis > 0.2:
                    Relations.append([i,j,WeightNose*WeightDis])
                    
    M = sparse.lil_matrix((len(Relations),n_source_vertices),dtype=np.float32)
    for i,t in enumerate(Relations):
        M[i,t[0]] = t[2]
        M[i,t[1]] = t[2]
        
    return M
        

def GetEdgeSet(faces):
    alledges=[]
    for face in faces:
        face = np.sort(face)
        alledges.append(tuple([face[0],face[1]]))
        alledges.append(tuple([face[0],face[2]]))
        alledges.append(tuple([face[1],face[2]]))
        
    edges = set(alledges)
    return edges
        

def EdgeDisMatrix(edges,vertices):
    n_source_edges = len(edges)
    n_source_vertices = vertices.shape[0]
    VectorToNose = vertices - vertices[_Nose_Index]
    DisToNose = np.linalg.norm(VectorToNose,axis=1)
    MaxDis = max(DisToNose)
    
    M2 = sparse.lil_matrix((n_source_edges, n_source_vertices), dtype=np.float32)
    
    for i,t in enumerate(edges):
        u = (DisToNose[t[0]] + DisToNose[t[1]])*0.6
        u2 = u/MaxDis + 0.2
        if u2 < 1:
            M2[i, t[0]] = -u2
            M2[i, t[1]] = u2
        else:
            M2[i, t[0]] = -0.9999999
            M2[i, t[1]] = 0.9999999
    return M2
