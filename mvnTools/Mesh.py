import pymesh
from mvnTools import Display as d
from skimage import measure
import wildmeshing as wm
import os
import meshio

def clean_mesh(mesh):
    print('\t - Cleaning mesh.')
    mesh, info = pymesh.remove_isolated_vertices(mesh)
    mesh, info = pymesh.remove_duplicated_faces(mesh)
    return mesh

def generate_surface(img_3d, iso=0, grad='descent', plot=True, offscreen=False):
    print('\t - Generating surface mesh.')
    verts, faces, normals, values = measure.marching_cubes_lewiner(img_3d, iso, gradient_direction=grad)
    mesh = pymesh.form_mesh(verts, faces)
    mesh = clean_mesh(mesh)
    verts = mesh.vertices
    faces = mesh.faces

    if plot:
        d.visualize_mesh((verts, faces), offscreen=offscreen)
    return verts, faces, mesh


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