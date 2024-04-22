import warnings
warnings.filterwarnings("ignore")
import gmsh
import numpy as np
import sys 

name = "rect"

### Parameters:
delta = 0.1
gamma_0 = 0.1
point_num = 160
mesh_size_ref = 0.005

show_mesh = False

### Roughness function:
def sin_rough(x):
   if x < delta or x > 1 - delta:
      return 1.0
   return 1.0 - (1-gamma_0)*(np.sin(2*np.pi*(x-delta)/(1 - 2.0*delta) - np.pi/2.0)+1.0) / 2.0

def rect_rough(x):
   if x < delta or x > 1 - delta:
      return 1.0
   if x < 2 * delta:
      x_shift = (x - delta) / (delta)
      return 1.0 - (1-gamma_0)*(-2 * (x_shift)**3 + 3 * x_shift**2)
   if x > 1 - 2 * delta:
      x_shift = (x - 1 + 2*delta) / (delta)
      return 1.0 - (1-gamma_0)*(1 + 2 * (x_shift)**3 - 3 * x_shift**2)
   return gamma_0

rough_fn = rect_rough #sin_rough

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

# Left and right need the same vertices for a periodic mesh
translation = [1, 0, 0, 1, 
               0, 1, 0, 0, 
               0, 0, 1, 0, 
               0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(1, [last_line], [last_line-2], translation)

### Mark subdomains
gmsh.model.addPhysicalGroup(2, [1], 1, name="Dummy")


gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_ref)

gmsh.model.mesh.generate(2)
gmsh.write("MeshCreation/2DMesh/cell_"+ name +"_gamma0_" + str(gamma_0) + ".msh")
# Launch the GUI to see the results:
if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()