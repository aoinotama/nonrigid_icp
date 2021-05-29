import numpy as np
import open3d as o3d
import copy
import sys


from icp import icp,draw_registration_result
from nricp import nonrigidIcp
from nricpReference import nonrigidIcpReference
from nricpWithReference import nonrigidIcpWithReference
from exportMatches import exportmatches

ShowMesh = False

folder = "Expression/Result/"

def ComputeNICP(sourceFileName,targetFileName,deformed_meshName,sourceFileNameRef,targetFilaNameRef):
    #Read source of referenced NICP model 
    sourcemeshRef = o3d.io.read_triangle_mesh(sourceFileNameRef)
    targetmeshRef = o3d.io.read_triangle_mesh(targetFilaNameRef)
    sourcemeshRef.compute_vertex_normals()
    targetmeshRef.compute_vertex_normals()

    sourcemesh = o3d.io.read_triangle_mesh(sourceFileName)
    targetmesh = o3d.io.read_triangle_mesh(targetFileName)
    sourcemesh.compute_vertex_normals()
    targetmesh.compute_vertex_normals()

    # first find rigid registration
    # guess for inital
    initial_guess = np.eye(4)
    affine_transformRef = icp(sourcemeshRef,targetmeshRef,initial_guess)
    
    initial_guess = np.eye(4)
    affine_transform = icp(sourcemesh,targetmesh,initial_guess)
    
    
    #creating a new mesh for non rigid transform estimation 
    refined_sourcemesh = copy.deepcopy(sourcemesh)
    refined_sourcemesh.transform(affine_transform)
    refined_sourcemesh.compute_vertex_normals()
    
    refined_sourcemeshRef = copy.deepcopy(sourcemeshRef)
    refined_sourcemeshRef.transform(affine_transformRef)
    refined_sourcemeshRef.compute_vertex_normals()
    
    #non rigid registration for good case

    matches,deformed_reference_mesh = nonrigidIcpReference(refined_sourcemeshRef,targetmeshRef)
    exportmatches(matches)
    print(matches.shape)
    # newmatches,deformed_mesh = nonrigidIcpReference(refined_sourcemesh,targetmesh,matches)
    newmatches,deformed_mesh = nonrigidIcpReference(targetmesh,refined_sourcemesh,matches,True)
    if ShowMesh:
        sourcemesh.paint_uniform_color([0.1, 0.9, 0.1])
        targetmesh.paint_uniform_color([0.9,0.1,0.1])
        deformed_mesh.paint_uniform_color([0.1,0.1,0.9])
        o3d.visualization.draw_geometries([targetmesh,deformed_mesh])
    o3d.io.write_triangle_mesh(deformed_meshName,deformed_mesh)
    o3d.io.write_triangle_mesh(deformed_meshName+"ref.obj",deformed_reference_mesh)

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        sourceFileNameRef = "exp3/Keeporder.obj"
        targetFilaNameRef = "exp3/Headv1.obj"
        sourceFileName = "exp3/Facev1.obj"
        targetFileName = "exp3/Headv1.obj"
        deformed_meshName = folder + "deformed_mesh.obj"
    if (len(sys.argv) == 3):
        sourceFileNameRef = "exp3/Keeporder.obj"
        targetFilaNameRef = "exp3/Headv1.obj"
    
        sourceFileName = sys.argv[1]
        targetFileName = sys.argv[2]
        deformed_meshName = folder + "deformed_mesh.obj"
    if (len(sys.argv) == 4):
        sourceFileNameRef = "exp3/Keeporder.obj"
        targetFilaNameRef = "exp3/Headv1.obj"
        sourceFileName = sys.argv[1]
        targetFileName = sys.argv[2]
        deformed_meshName = sys.argv[3]
    if (len(sys.argv) == 6):
        sourceFileName = sys.argv[1]
        targetFileName = sys.argv[2]
        deformed_meshName = sys.argv[3]
        sourceFileNameRef = sys.argv[4]
        targetFilaNameRef = sys.argv[5]
    
    ComputeNICP(sourceFileName,targetFileName,deformed_meshName,sourceFileNameRef,targetFilaNameRef)