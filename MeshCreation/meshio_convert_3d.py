import meshio

def create_mesh(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, 
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh

eps = 0.1

msh=meshio.read("MeshCreation/3DMesh/fluid_domain" + str(eps) + ".msh")
tetraeder_mesh = create_mesh(msh, "tetra")
meshio.write("MeshCreation/3DMesh/fluid_domain" + str(eps) + ".xdmf", tetraeder_mesh)

msh=meshio.read("MeshCreation/3DMesh/grind_domain" + str(eps) + ".msh")
tetraeder_mesh = create_mesh(msh, "tetra")
meshio.write("MeshCreation/3DMesh/solid_domain" + str(eps) + ".xdmf", tetraeder_mesh)

msh=meshio.read("MeshCreation/3DMesh/ref_cell.msh")
tetraeder_mesh = create_mesh(msh, "tetra")
meshio.write("MeshCreation/3DMesh/ref_cell.xdmf", tetraeder_mesh)