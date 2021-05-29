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
    try:
        factor = cholesky_AAt(M.T)
        return factor(M.T.dot(b)).toarray()
    except:
        print("CholeskySolve failed, Try sparse.lsqr to solve")
        print(b.shape)
        print(M.shape)
        # Divide b into 3 part
        b0 = b[:,0].toarray()
        b1 = b[:,1].toarray()
        b2 = b[:,2].toarray()
        print(b0.shape)
        x0 = sparse.linalg.lsqr(M,b0)[0]
        print(x0.shape)
        x1 = sparse.linalg.lsqr(M,b1)[0]
        x2 = sparse.linalg.lsqr(M,b2)[0]
        x = np.array([x0,x1,x2])
        x = x.transpose()
        print(x.shape)
        return x

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
        if i %100 ==0:
            # print(i)
            pass
        for j in range(i+1,len(vertices)):
            if (DisToNose[i] < 0.8 * MaxDis):
                WeightNose = 1 - DisToNose[i]/MaxDis
                VertDis = np.linalg.norm(vertices[i]-vertices[j])
                WeightDis = 1 - VertDis/MaxDis
                if WeightDis < 0:
                    WeightDis = 0
                # Discard small Weight to Reduce 
                if WeightNose * WeightDis > 0.8:
                    Relations.append([i,j,WeightNose*WeightDis])
                    
    print(len(Relations))
    M = sparse.lil_matrix((len(Relations),n_source_vertices),dtype=np.float32)
    for i,t in enumerate(Relations):
        M[i,t[0]] = -t[2]
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


def ReverseMatches(matches,source,target):
    for i in matches:
        # print(i)
        pass
    source_vertices = np.array(source.vertices)
    target_vertices = np.array(target.vertices)
    n_source = source_vertices.shape[0]
    n_target = target_vertices.shape[0]
    #n_source = len(source)
    #n_target = len(target)
    reverse_matches = np.ndarray([n_target],dtype=np.int32)
    reverse_matches.fill(-1)
    L = [12,22,23,24,26,39,41,357,359,360,361,364,365,366,370,371,372,374,381,382,383,385,386,387,389,392,393,394,399,400,403,405,407,409,411,415,417,419,421,424,442,443,455,465,466,467,476,478,774,776,777,778,781,782,783,787,788,789,791,796,797,798,800,801,802,803,806,807,808,813,814,816,818,820,822,824,827,829,831,833,1767,1772,1823,1827,1863,1864,1866,1867,1910,1913,1964,1967,2002,2005,2073,2076,2077,2081,2083,2086,2095,2096,2129,2131,2162,2164,2205,2208,2266,2269,2298,2301,2312,2313,2330,2333,2419,2421,2423,2424,2460,2463,2509,2510,2535,2536,2580,2581,2589,2590,2591,2592,2658,2659,2817,2818,3116,3120,3156,3158,3165,3168,3212,3213,3254,3257,3258,3261,3332,3333,3375,3377,3382,3383,3389,3392,3427,3428,3447,3448]
    for i in range(0,len(matches)):
        print(str(i) + ": " + str(matches[i]))
        if (i + 1) in L:            
            reverse_matches[matches[i]] = i
    return reverse_matches
        
def GetMisMatches(matches):
    Mismatches = []
    for i in range(0,len(matches)):
        if matches[i] == -1:
            Mismatches.append(i)
            # print(i)
        #else:
        #    _matches[i] = 1
    Mismatches = np.array(Mismatches,dtype=np.int32)
    return Mismatches

def NICPsolve(source_vertices,source_faces,target_vertices,target_face,alpha_stiffness = 1.0):

    #calculating edge info
    sourcemesh_faces = source_faces
    knnsearch = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)
    
    alledges=[]
    for face in sourcemesh_faces:
        face = np.sort(face)
        alledges.append(tuple([face[0],face[1]]))
        alledges.append(tuple([face[0],face[2]]))
        alledges.append(tuple([face[1],face[2]]))
        
    edges = set(alledges)
    n_source_edges = len(edges)
    n_source_verts = source_vertices.shape[0]

    M = sparse.lil_matrix((n_source_edges, n_source_verts), dtype=np.float32)
    print(M)
    print("M printed")
    for i, t in enumerate(edges):
        M[i, t[0]] = -1
        M[i, t[1]] = 1
    
    
    G = np.diag([1, 1, 1, gamma]).astype(np.float32)
    
    kron_M_G = sparse.kron(M, G)
    print(kron_M_G)
    print("kron_G_printed")

    # X for transformations and D for vertex info in sparse matrix
    # using lil_matrix because chinging sparsity in csr is expensive 
    #Equation -> 8
    D = sparse.lil_matrix((n_source_verts,n_source_verts*4), dtype=np.float32)
    print(D)
    print("DDDDD")
    j_=0
    for i in range(n_source_verts):
        D[i,j_:j_+3]=source_vertices[i,:]
        D[i,j_+3]=1
        j_+=4

    print(D)

    #AFFINE transformations stored in the 4n*3 format
    X_= np.concatenate((np.eye(3),np.array([[0,0,0]])),axis=0)
    X = np.tile(X_,(n_source_verts,1))
    
   
    # 
    wVec = np.ones((n_source_verts,1))
    print(wVec)
    
    vertsTransformed = D*X
    
    distances, indices = knnsearch.kneighbors(vertsTransformed)

    indices = indices.squeeze()
            
    matches = target_vertices[indices]
            
    #rigtnow setting threshold manualy, but if we have and landmark info we could set here
    mismatches = np.where(distances>15)[0]

    # setting weights of false mathces to zero   
    wVec[mismatches] = 0
            
                
    # Equation  12
    # E(X) = ||AX-B||^2
            
    U = wVec*matches
            
    A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * kron_M_G,   D.multiply(wVec) ]))
    
    print(A.toarray())
    print("A*****A")
            
    B = sparse.lil_matrix((4 * n_source_edges + n_source_verts, 3), dtype=np.float32)
            
    B[4 * n_source_edges: (4 * n_source_edges +n_source_verts), :] = U
            
    print(B.toarray()) 
    print("B******B")
    X = choleskySolveM(A, B)

    return X
    

def PrintInfo(A,B,X,U,D,wVec,alpha_stiffness,kron_M_G):
    # First Part: Print Transformation
    # print X && target & deformed 
    print("Transformation matrix is")
    print(X)
    
    print("Target is")
    print(U)
    
    # print("Deformed is")
    # print(D)
    
    print("Deformed is")
    print(D * X)
    
    print("Deformed match")
    print(D.multiply(wVec)*X)
    
    