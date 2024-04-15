import meshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.points = out_mesh.points[:, :2]
    return out_mesh

eps = 0.1

msh=meshio.read("MeshCreation/2DMesh/fluid_domain" + str(eps) + ".msh")
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/fluid_domain" + str(eps) + ".xdmf", triangle_mesh)

msh=meshio.read("MeshCreation/2DMesh/grind_domain" + str(eps) + ".msh")
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/grind_domain" + str(eps) + ".xdmf", triangle_mesh)

msh=meshio.read("MeshCreation/2DMesh/ref_cell.msh")
triangle_mesh = create_mesh(msh, "triangle", True)
meshio.write("MeshCreation/2DMesh/ref_cell.xdmf", triangle_mesh)