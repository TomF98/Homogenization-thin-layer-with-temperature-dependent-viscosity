import meshio

"""
Converts the gmsh files to fenics compatible files.
"""

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.points = out_mesh.points[:, :2]
    return out_mesh

eps = 0.1
name = "sin"
gamma_0 = 0.5

msh=meshio.read("MeshCreation/2DMesh/"+ str(name) +"_fluid_domain_gamma0_" + str(gamma_0) + 
               "_eps_" + str(eps) +'.msh')
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/XDMF/"+ str(name) +"_fluid_domain_gamma0_" + str(gamma_0) + 
               "_eps_" + str(eps) +'.xdmf', triangle_mesh)

msh=meshio.read("MeshCreation/2DMesh/"+ str(name) +"_solid_domain_gamma0_" + str(gamma_0) + 
               "_eps_" + str(eps) +'.msh')
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/XDMF/"+ str(name) +"_solid_domain_gamma0_" + str(gamma_0) + 
               "_eps_" + str(eps) +'.xdmf', triangle_mesh)

msh=meshio.read("MeshCreation/2DMesh/cell_"+ name +"_gamma0_" + str(gamma_0) + ".msh")
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/XDMF/cell_"+ name +"_gamma0_" + str(gamma_0) + ".xdmf", 
             triangle_mesh)