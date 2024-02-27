import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys 

### Parameters:
gamma_0 = 0.5
point_num = 80
mesh_size_ref = 0.005

show_mesh = False

### Roughness function:
def sin_rough(x):
   return min(1 + (1-gamma_0)*np.sin(2*np.pi*x + np.pi/2.0), 1.0)

def triangle_rough(x):
   x_s = x + 1/2.0
   osci = x_s%1 - int(x_s%1 + 0.5)
   return min(1.0 + (1-gamma_0) * (2*abs(2*osci)-1.0), 1.0)

rough_fn = sin_rough

### Start domain creation:
gmsh.initialize()
gmsh.model.add("Domain")

point_list = np.zeros(point_num+3, dtype=np.int16)

for i in range(point_num+1):
    x = 1/point_num * i
    point_list[i] = gmsh.model.occ.addPoint(x, rough_fn(x), 0.0)

# Add bottom corners:
point_list[-2] = gmsh.model.occ.addPoint(1, 0.0, 0.0)
point_list[-1] = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)

# Create edges
for i in range(len(point_list)-1):
   gmsh.model.occ.addLine(point_list[i], point_list[i+1])

last_line = gmsh.model.occ.addLine(point_list[-1], point_list[0])

# Create 2D object_
loop = gmsh.model.occ.addCurveLoop(point_list)
fluid_layer = gmsh.model.occ.addPlaneSurface([loop])

gmsh.model.occ.synchronize()

translation = [1, 0, 0, 1, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(1, [last_line], [last_line-2], translation)

### Mark subdomains
gmsh.model.addPhysicalGroup(2, [1], 1, name="Dummy")


gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_ref)

gmsh.model.mesh.generate(2)
gmsh.write('MeshCreation/2DMesh/ref_cell.msh')
# Launch the GUI to see the results:
if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()