import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sksparse.cholmod import cholesky_AAt
import open3d as o3d
import copy

def choleskySolve(M, b):

    factor = cholesky_AAt(M.T)
    return factor(M.T.dot(b)).toarray()




Debug=False
DebugByStep=True
DebugShowStep=False
DebugExportStep=True

folder = "Expression/Reference/"

normalWeighting=False
gamma = 1
alphas = np.linspace(2000,1,20)

def nonrigidIcpReference(sourcemesh,targetmesh):
    
    refined_sourcemesh = copy.deepcopy(sourcemesh)
    #obtain vertices
    target_vertices = np.array(targetmesh.vertices)
    source_vertices = np.array(refined_sourcemesh.vertices)
    #num of source mesh vertices 
    n_source_verts = source_vertices.shape[0]
    
    #normals again for refined source mesh and target mesh
    source_mesh_normals = np.array(refined_sourcemesh.vertex_normals)
    target_mesh_normals = np.array(targetmesh.vertex_normals)


    knnsearch = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)

    sourcemesh_faces = np.array(sourcemesh.triangles)
    
    #calculating edge info
    alledges=[]
    for face in sourcemesh_faces:
        face = np.sort(face)
        alledges.append(tuple([face[0],face[1]]))
        alledges.append(tuple([face[0],face[2]]))
        alledges.append(tuple([face[1],face[2]]))
        
    edges = set(alledges)
    n_source_edges = len(edges)

    try:
        print(source_vertices[114])
        print(source_vertices[113])
    except:
        print("Can not print nose")
    
    # Calculate Distane to Nose
    VectorToNose = source_vertices - source_vertices[114]
    DisToNose = np.linalg.norm(VectorToNose,axis=1)
    MaxDis = max(DisToNose)   
    
    #Set Vertice Stiffness Weight Acccording to the distances
    M2 = sparse.lil_matrix((n_source_edges, n_source_verts), dtype=np.float32)
    
    for i,t in enumerate(edges):
        u = (DisToNose[t[0]] + DisToNose[t[1]])*0.6
        u2 = u/MaxDis + 0.2
        if u2 < 1:
            M2[i, t[0]] = -u2
            M2[i, t[1]] = u2
        else:
            M2[i, t[0]] = -1
            M2[i, t[1]] = 1
            
       
    
    
    
    
    M = sparse.lil_matrix((n_source_edges, n_source_verts), dtype=np.float32)
    
    for i, t in enumerate(edges):
        M[i, t[0]] = -1
        M[i, t[1]] = 1
    
    
    G = np.diag([1, 1, 1, gamma]).astype(np.float32)
    kron_M_G2 = sparse.kron(M2,G) 
    
    kron_M_G = sparse.kron(M, G)

    print(kron_M_G)
    print(kron_M_G2)
    

    # X for transformations and D for vertex info in sparse matrix
    # using lil_matrix because chinging sparsity in csr is expensive 
    #Equation -> 8
    D = sparse.lil_matrix((n_source_verts,n_source_verts*4), dtype=np.float32)
    j_=0
    for i in range(n_source_verts):
        D[i,j_:j_+3]=source_vertices[i,:]
        D[i,j_+3]=1
        j_+=4



    #AFFINE transformations stored in the 4n*3 format
    X_= np.concatenate((np.eye(3),np.array([[0,0,0]])),axis=0)
    X = np.tile(X_,(n_source_verts,1))


    if Debug:
        targetmesh.paint_uniform_color([0.9,0.1,0.1])
        refined_sourcemesh.paint_uniform_color([0.1,0.1,0.9])
        o3d.visualization.draw_geometries([targetmesh,refined_sourcemesh])

    
    
    
    
    if normalWeighting:
        n_source_normals = len(source_mesh_normals) #will be equal to n_source_verts
        DN = sparse.lil_matrix((n_source_normals,n_source_normals*4), dtype=np.float32)
        j_=0
        for i in range(n_source_normals):
            DN[i,j_:j_+3]=source_mesh_normals[i,:]
            DN[i,j_+3]=1
            j_+=4




    for num_,alpha_stiffness in enumerate(alphas):
        
        print("step- {}/20".format(num_))
        
        for i in range(3):
        
            alpha_stiffness = alphas[0]
            
            print(alpha_stiffness)
            
            wVec = np.ones((n_source_verts,1))
            
            vertsTransformed = D*X
            
            distances, indices = knnsearch.kneighbors(vertsTransformed)
            
            indices = indices.squeeze()
            
            matches = target_vertices[indices]
            
            #rigtnow setting threshold manualy, but if we have and landmark info we could set here
            mismatches = np.where(distances>15)[0]
            
            
            if normalWeighting:
                normalsTransformed = DN*X
                corNormalsTarget = target_mesh_normals[indices]
                crossNormals = np.cross(corNormalsTarget, normalsTransformed)
                crossNormalsNorm = np.sqrt(np.sum(crossNormals**2,1))
                dotNormals = np.sum(corNormalsTarget*normalsTransformed,1)
                angles =np.arctan(dotNormals/crossNormalsNorm)
                wVec = wVec *(angles<np.pi/4).reshape(-1,1)
                
                
                
    
            #setting weights of false mathces to zero   
            wVec[mismatches] = 0
                
            #Equation  12
            #E(X) = ||AX-B||^2
            
            U = wVec*matches
            
            kron_M_G_Face = kron_M_G # -  ((num_+1)/20) * kron_M_G2
            
            A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * kron_M_G_Face,   D.multiply(wVec) ]))
            
            B = sparse.lil_matrix((4 * n_source_edges + n_source_verts, 3), dtype=np.float32)
            
            B[4 * n_source_edges: (4 * n_source_edges +n_source_verts), :] = U
            
            X = choleskySolve(A, B)
        
            try:
                ErrorStiffness = alpha_stiffness * kron_M_G_Face * X
                ErrorFace = kron_M_G_Face * X
                try:
                    # print(ErrorStiffness.shape)
                    pass
                except:
                    print("Can not print ErrorStiffness's shape")
                try:
                    pass
                    # print(len(ErrorStiffness))
                except:
                    print("Can not print length of ErrorStiffness")
                try: 
                    pass
                    # print(type(ErrorStiffness))
                except:
                    print("Can not Get the type of ErrorStiffness")
                try:
                    temp = ErrorStiffness - B[0:4 * n_source_edges, :]
                    try:
                        ErrorS = np.linalg.norm(temp)
                        ErrorFace = np.linalg.norm(ErrorFace - B[0:4 * n_source_edges, :])
                        # print("Stiffness Error is  ", end=" ")
                        # print(ErrorS)
                        # print("Face Error is", end=" ")
                        print(ErrorFace)
                    except Exception as er1:
                        print("Can not print Error")
                        print(er1);
                except Exception as er2:
                    print("Can not Calculate Erorr")
                    print(er2) 
            except:
                print("Can not calculate ErrorStiffness")
            
            try:
                ErrorVerticesMatrix = D.multiply(wVec)*X - U
                try:
                    ErrorVertices = np.linalg.norm(ErrorVerticesMatrix)
                    try:
                        # print("Vertices Error is  ", end=" ")
                        # print(ErrorVertices)
                        pass
                    except:
                        print("Can not Print out ErrorVertices")
                except:
                    print("Can not Calculate Error Vertices")
            except:
                print("Can not Calculate ErrorVerticesMatrix")
                print("Can not calculate ErrorStiffness")
            
            try:
                # print("Total Error is", end=' ')
                # print(np.linalg.norm(A*X-B))
                pass
            except:
                print("Can not Calculate Total Error")
             
            
        if DebugByStep:
            #Extra Part To Export Mesh for every Step
            
            vertsTransformed_export = D*X;

            refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed_export)
    
            #project source on to template
            matcheindices_export = np.where(wVec > 0)[0]
            vertsTransformed_export[matcheindices_export]=matches[matcheindices_export]
            refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed_export)
                
            if DebugShowStep:
                #print Out Result After each step
                targetmesh.paint_uniform_color([0.9,0.1,0.1])             
                refined_sourcemesh.paint_uniform_color([0.1,0.1,0.9])
                o3d.visualization.draw_geometries([targetmesh,refined_sourcemesh])   
            if DebugExportStep:
                print(folder + "deformed_mesh" + "step{}/20".format(num_)+".obj")
                
                o3d.io.write_triangle_mesh(folder + "deformed_mesh" + "step{}20".format(num_)+".obj",refined_sourcemesh)
            
    vertsTransformed = D*X;

    refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed)
    
    #project source on to template
    matcheindices = np.where(wVec > 0)[0]
    vertsTransformed[matcheindices]=matches[matcheindices]
    refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed)




    return indices;
