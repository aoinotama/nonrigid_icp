import numpy as np
import open3d as o3d
import copy
import sys

from icp import icp,draw_registration_result
from nricp import nonrigidIcp


def ComputeNICP(sourceFileName,targetFileName,deformed_meshName):
    #read source file
    
    sourcemesh = o3d.io.read_triangle_mesh(sourceFileName)
    targetmesh = o3d.io.read_triangle_mesh(targetFileName)
    sourcemesh.compute_vertex_normals()
    targetmesh.compute_vertex_normals()
    





    #first find rigid registration
    # guess for inital transform for icp
    initial_guess = np.eye(4)
    affine_transform = icp(sourcemesh,targetmesh,initial_guess)


    #creating a new mesh for non rigid transform estimation 
    refined_sourcemesh = copy.deepcopy(sourcemesh)
    refined_sourcemesh.transform(affine_transform)
    refined_sourcemesh.compute_vertex_normals()


    #non rigid registration
    deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)



    sourcemesh.paint_uniform_color([0.1, 0.9, 0.1])
    targetmesh.paint_uniform_color([0.9,0.1,0.1])
    deformed_mesh.paint_uniform_color([0.1,0.1,0.9])
    o3d.visualization.draw_geometries([targetmesh,deformed_mesh])
    o3d.io.write_triangle_mesh(deformed_meshName,deformed_mesh)

    
    





if __name__ == "__main__":
    if (len(sys.argv) == 1):
        sourceFileName = "exp3/Facev1.obj"
        targetFileName = "exp3/Headv1.obj"
        deformed_meshName = "deformed_mesh.obj"
    if (len(sys.argv) == 3):
        sourceFileName = sys.argv[1]
        targetFileName = sys.argv[2]
        deformed_meshName = "deformed_mesh.obj"
    if (len(sys.argv) == 4):
        sourceFileName = sys.argv[1]
        targetFileName = sys.argv[2]
        deformed_meshName = sys.argv[3]
    ComputeNICP(sourceFileName,targetFileName,deformed_meshName)

