import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys 

### Parameters:
L, H = 1, 1
eps = 0.01
gamma_0 = 0.5
point_num = int(L/eps * 40) # for discrete interface, points per eps section

fluid_marker = 0
grinding_marker = 1

mesh_size_max = 0.025
mesh_size_min = 0.05 * eps
show_mesh = False
save_fluid = True

### Roughness function:
def sin_rough(x):
   return min(1 + (1-gamma_0)*np.sin(2*np.pi*x/eps + np.pi/2.0), 1.0)

def triangle_rough(x):
   x_s = x + eps/2.0
   osci = x_s%eps / eps - int(x_s%eps / eps + 0.5)
   return min(1.0 + (1-gamma_0) * (2*abs(2*osci)-1.0), 1.0)

rough_fn = sin_rough

### Start domain creation:
gmsh.initialize()
gmsh.model.add("Domain")

point_list = np.zeros(point_num+3, dtype=np.int16)

for i in range(point_num+1):
    x = L/point_num * i
    point_list[i] = gmsh.model.occ.addPoint(x, eps*rough_fn(x), 0.0)

# Add bottom corners:
point_list[-2] = gmsh.model.occ.addPoint(L, 0.0, 0.0)
point_list[-1] = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)

# Create edges
for i in range(len(point_list)-1):
   gmsh.model.occ.addLine(point_list[i], point_list[i+1])

gmsh.model.occ.addLine(point_list[-1], point_list[0])

# Create 2D object_
loop = gmsh.model.occ.addCurveLoop(point_list)
fluid_layer = gmsh.model.occ.addPlaneSurface([loop])

grind_wheel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

gmsh.model.occ.synchronize()
omega = gmsh.model.occ.fragment([[2, grind_wheel]], [[2, fluid_layer]])

gmsh.model.occ.synchronize()
### Mark subdomains
if save_fluid:
    gmsh.model.addPhysicalGroup(2, [1], fluid_marker, name="Fluid")
else:
    gmsh.model.addPhysicalGroup(2, [2], grinding_marker, name="Wheel")


gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_max)

# ### Higher resolution around interface
interface_area = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(interface_area, "Thickness", 0.1)
gmsh.model.mesh.field.setNumber(interface_area, "VIn", mesh_size_min)
gmsh.model.mesh.field.setNumber(interface_area, "VOut", mesh_size_max)
gmsh.model.mesh.field.setNumber(interface_area, "XMax", L + 0.1)
gmsh.model.mesh.field.setNumber(interface_area, "XMin", -0.1)
gmsh.model.mesh.field.setNumber(interface_area, "ZMax", 1)
gmsh.model.mesh.field.setNumber(interface_area, "ZMin", -1)
gmsh.model.mesh.field.setNumber(interface_area, "YMax", 3*eps)
gmsh.model.mesh.field.setNumber(interface_area, "YMin", -0.1)
gmsh.model.mesh.field.setAsBackgroundMesh(interface_area)

gmsh.model.mesh.generate(2)
if save_fluid:
    gmsh.write('MeshCreation/2DMesh/fluid_domain' + str(eps) + '.msh')
else:
    gmsh.write('MeshCreation/2DMesh/grind_domain' + str(eps) + '.msh')
# Launch the GUI to see the results:
if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()