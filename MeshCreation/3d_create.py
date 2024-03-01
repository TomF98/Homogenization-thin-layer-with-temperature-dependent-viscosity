import gmsh
import numpy as np
import sys

### Parameters:
L, H, B = 1, 1, 1
eps = 0.1
gamma_0 = 0.5
point_num = int(L/eps * 32) # for discrete interface, points per eps section

fluid_marker = 0
grinding_marker = 1

mesh_size_max = 0.05
mesh_size_min = 0.1 * eps

show_mesh = False
save_fluid = False

### Roughness function:
def sin_rough(x, y):
   sin_term = np.sin(2*np.pi*x/eps) * np.sin(2*np.pi*y/eps)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)

def sin_groove_x(x, y):
   sin_term = np.sin(2*np.pi*x/eps + np.pi/2.0)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)

def sin_groove_y(x, y):
   sin_term = np.sin(2*np.pi*y/eps + np.pi/2.0)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)


rough_fn = sin_groove_x

### Start domain creation
## Build grid with height of roughness:
x_coords = np.linspace(0, L, point_num)
y_coords = np.linspace(0, B, point_num)
coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
coords = np.column_stack((coords, eps * rough_fn(coords[:, :1], coords[:, 1:])))


## Start gmsh
gmsh.initialize()
gmsh.model.add("Domain")

## We will create the terrain surface mesh from input data points:
## Helper function to return a node tag given two indices i
## (x direction) and j (y direction):
def tag(i, j):
    return point_num * i + j + 1

# The tags of the corresponding nodes:
nodes = []
# The connectivities of the triangle elements (3 node tags per triangle) on the
# interface:
tris = []
# The connectivities of the line elements on the 4 boundaries. So lin[0] contains the 
# boundary line at position j==0==y (2 node tags for each line element):
lin = [[], [], [], []]
# The point elements on the 4 corners (1 node tag for each point element):
pnt = [tag(0, 0), tag(point_num-1, 0), 
       tag(point_num-1, point_num-1), tag(0, point_num-1)]

for i in range(point_num):
    for j in range(point_num):
        node_tag = tag(i, j)
        # Add tag and coordinate of the point:
        nodes.append(tag(i, j))

        # Once we have created the first line we can connect triangles to the previous points:
        if i > 0 and j > 0:
            tris.extend([tag(i - 1, j - 1), tag(i, j - 1), tag(i - 1, j)])
            tris.extend([tag(i, j - 1), tag(i, j), tag(i - 1, j)])

        # Same idea for the boundary lines
        if (i == 0 or i == point_num-1) and j > 0:
            lin[3 if i == 0 else 1].extend([tag(i, j - 1), tag(i, j)])
        if (j == 0 or j == point_num-1) and i > 0:
            lin[0 if j == 0 else 2].extend([tag(i - 1, j), tag(i, j)])

# Create 4 discrete points for the 4 corners of the domain surface:
for i in range(4):
    # Input is dimension of entity (0 = Point) and some tag/index
    gmsh.model.addDiscreteEntity(0, i + 1)
# Input is the previous tag/index and the new coordinates
gmsh.model.setCoordinates(1, *coords[0, :3])
gmsh.model.setCoordinates(2, *coords[tag(point_num-1, 0) - 1, :3])
gmsh.model.setCoordinates(3, *coords[tag(point_num-1, point_num-1) - 1, :3])
gmsh.model.setCoordinates(4, *coords[tag(0,           point_num-1) - 1, :3])

# Create 4 discrete bounding curves, with their boundary points:
for i in range(4):
    # Again dimension of entity (1 = Line), some tag/index (is unique per dimension
    # so we can repeat 1,2,3,4) and the points that should be connected.
    # For the points we have to use the corresponding tags
    gmsh.model.addDiscreteEntity(1, i + 1, [i + 1, i + 2 if i < 3 else 1])

# Create one discrete surface, with its bounding curves:
gmsh.model.addDiscreteEntity(2, 1, [1, 2, -3, -4]) # 2 == Surface given by four lines

# Add all the nodes on the surface (for simplicity... see below):
gmsh.model.mesh.addNodes(2, 1, nodes, coords.flatten())

# Add point elements on the 4 points, line elements on the 4 curves, and
# triangle elements on the surface:
for i in range(4):
    # Type 15 for point elements:
    gmsh.model.mesh.addElementsByType(i + 1, 15, [], [pnt[i]])
    # Type 1 for 2-node line elements:
    gmsh.model.mesh.addElementsByType(i + 1, 1, [], lin[i])

# Type 2 for 3-node triangle elements:
gmsh.model.mesh.addElementsByType(1, 2, [], tris)

# Reclassify the nodes on the curves and the points (since we put them all on
# the surface before with `addNodes' for simplicity)
gmsh.model.mesh.reclassifyNodes()

# Create a geometry for the discrete curves and surfaces, so that we can remesh
# them later on:
gmsh.model.mesh.createGeometry()

# First add wheel domain:
p1 = gmsh.model.geo.addPoint(*coords[0, :2], H, meshSize=mesh_size_max)
p2 = gmsh.model.geo.addPoint(*coords[tag(point_num-1, 0) - 1, :2], H, 
                             meshSize=mesh_size_max)
p3 = gmsh.model.geo.addPoint(*coords[tag(point_num-1, point_num-1) - 1, :2], 
                             H, meshSize=mesh_size_max)
p4 = gmsh.model.geo.addPoint(*coords[tag(0, point_num-1) - 1, :2], 
                             H, meshSize=mesh_size_max)
c1 = gmsh.model.geo.addLine(p1, p2)
c2 = gmsh.model.geo.addLine(p2, p3)
c3 = gmsh.model.geo.addLine(p3, p4)
c4 = gmsh.model.geo.addLine(p4, p1)
c10 = gmsh.model.geo.addLine(p1, 1)
c11 = gmsh.model.geo.addLine(p2, 2)
c12 = gmsh.model.geo.addLine(p3, 3)
c13 = gmsh.model.geo.addLine(p4, 4)
ll1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
s1 = gmsh.model.geo.addPlaneSurface([ll1])
ll3 = gmsh.model.geo.addCurveLoop([c1, c11, -1, -c10])
s3 = gmsh.model.geo.addPlaneSurface([ll3])
ll4 = gmsh.model.geo.addCurveLoop([c2, c12, -2, -c11])
s4 = gmsh.model.geo.addPlaneSurface([ll4])
ll5 = gmsh.model.geo.addCurveLoop([c3, c13, 3, -c12])
s5 = gmsh.model.geo.addPlaneSurface([ll5])
ll6 = gmsh.model.geo.addCurveLoop([c4, c10, 4, -c13])
s6 = gmsh.model.geo.addPlaneSurface([ll6])
sl1 = gmsh.model.geo.addSurfaceLoop([s1, s3, s4, s5, s6, 1])
v1 = gmsh.model.geo.addVolume([sl1])
gmsh.model.geo.synchronize() # add the rough interface and this volume together

# # Next add fluid domain:
p1_s = gmsh.model.geo.addPoint(*coords[0, :2], 0.0, meshSize=mesh_size_min)
p2_s = gmsh.model.geo.addPoint(*coords[tag(point_num-1, 0) - 1, :2], 
                               0.0, meshSize=mesh_size_min)
p3_s = gmsh.model.geo.addPoint(*coords[tag(point_num-1, point_num-1) - 1, :2], 
                               0.0, meshSize=mesh_size_min)
p4_s = gmsh.model.geo.addPoint(*coords[tag(0, point_num-1) - 1, :2], 
                               0.0, meshSize=mesh_size_min)
c1_s = gmsh.model.geo.addLine(p1_s, p2_s)
c2_s = gmsh.model.geo.addLine(p2_s, p3_s)
c3_s = gmsh.model.geo.addLine(p3_s, p4_s)
c4_s = gmsh.model.geo.addLine(p4_s, p1_s)
c10_s = gmsh.model.geo.addLine(p1_s, 1)
c11_s = gmsh.model.geo.addLine(p2_s, 2)
c12_s = gmsh.model.geo.addLine(p3_s, 3)
c13_s = gmsh.model.geo.addLine(p4_s, 4)
ll1_s = gmsh.model.geo.addCurveLoop([c1_s, c2_s, c3_s, c4_s])
s1_s = gmsh.model.geo.addPlaneSurface([ll1_s])
ll3_s = gmsh.model.geo.addCurveLoop([c1_s, c11_s, -1, -c10_s])
s3_s = gmsh.model.geo.addPlaneSurface([ll3_s])
ll4_s = gmsh.model.geo.addCurveLoop([c2_s, c12_s, -2, -c11_s])
s4_s = gmsh.model.geo.addPlaneSurface([ll4_s])
ll5_s = gmsh.model.geo.addCurveLoop([c3_s, c13_s, 3, -c12_s])
s5_s = gmsh.model.geo.addPlaneSurface([ll5_s])
ll6_s = gmsh.model.geo.addCurveLoop([c4_s, c10_s, 4, -c13_s])
s6_s = gmsh.model.geo.addPlaneSurface([ll6_s])
sl1_s = gmsh.model.geo.addSurfaceLoop([s1_s, s3_s, s4_s, s5_s, s6_s, 1])
v2 = gmsh.model.geo.addVolume([sl1_s])
gmsh.model.geo.synchronize()


gmsh.model.mesh.setSize([(0, 1), (0, 2), (0, 3), (0, 4)], mesh_size_min)
gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size_min)
gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_max)

### Mark subdomains
if save_fluid:
    gmsh.model.addPhysicalGroup(3, [2], fluid_marker, name="Fluid")
else:
    gmsh.model.addPhysicalGroup(3, [1], grinding_marker, name="Wheel")

gmsh.model.mesh.generate(3)

if save_fluid:
    gmsh.write('MeshCreation/3DMesh/fluid_domain' + str(eps) + '.msh')
else:
    gmsh.write('MeshCreation/3DMesh/grind_domain' + str(eps) + '.msh')

if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()