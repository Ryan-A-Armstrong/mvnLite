import pymesh
from mvnTools import Display as d
from skimage import measure
import wildmeshing as wm
import os
import meshio

def clean_mesh(mesh, connected=True):
    print('\t - Cleaning mesh (this may take a moment)')
    mesh, info = pymesh.remove_isolated_vertices(mesh)
    mesh, info = pymesh.remove_duplicated_vertices(mesh)
    mesh, info = pymesh.remove_degenerated_triangles(mesh)
    mesh, info = pymesh.remove_duplicated_faces(mesh)

    if connected:
        mesh_list = pymesh.separate_mesh(mesh, 'auto')
        max_verts = 0
        print('Total number of meshes (ideally 1): %d' %  len(mesh_list))
        for mesh_obj in mesh_list:
            verts = mesh_obj.num_vertices
            if verts > max_verts:
                max_verts = verts
                mesh = mesh_obj

    return mesh

def generate_surface(img_3d, iso=0, grad='descent', plot=True, offscreen=False, connected=True, clean=True):
    print('\t - Generating surface mesh')
    verts, faces, normals, values = measure.marching_cubes_lewiner(img_3d, iso, gradient_direction=grad)
    mesh = pymesh.form_mesh(verts, faces)
    if clean:
        mesh = clean_mesh(mesh, connected=connected)
    verts = mesh.vertices
    faces = mesh.faces

    if plot:
        d.visualize_mesh((verts, faces), offscreen=offscreen)
    return mesh


def generate_lumen_tetmsh(path_to_surface_obj, path_to_volume_msh='', removeOBJ=False):
    if path_to_volume_msh == '':
        path_to_volume_msh = path_to_volume_msh.split('.')[0] + '.msh'

    wm.Tetrahedralizer(path_to_surface_obj, path_to_volume_msh)

    if removeOBJ:
        os.remove(path_to_surface_obj)

def create_ExodusII_file(path_to_msh, path_to_e='', removeMSH=False):
    if path_to_msh == '':
        path_to_e = path_to_msh.split('.')[0] + '.e'

    meshio.convert(path_to_msh, path_to_e)

    if removeMSH:
        os.remove(path_to_msh)