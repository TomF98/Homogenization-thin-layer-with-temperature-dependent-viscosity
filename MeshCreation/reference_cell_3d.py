import gmsh
import numpy as np
import sys

### Parameters:
gamma_0 = 0.1
point_num = 80
mesh_size_ref = 0.02

show_mesh = True

### Roughness function:
def sin_double(x, y):
    return 1 - (1-gamma_0)*(np.cos(2*np.pi*(y - 0.5)**2) * np.cos(2*np.pi*(x - 0.25)))**2

def sin_rough(x, y):
   sin_term = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)

def sin_groove_x(x, y):
   sin_term = np.sin(2*np.pi*x + np.pi/2.0)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)

def sin_groove_y(x, y):
   sin_term = np.sin(2*np.pi*y + np.pi/2.0)
   return np.clip(1 + (1-gamma_0)*sin_term, 0.0, 1.0)


rough_fn = sin_double

### Build cell
gmsh.initialize()
gmsh.model.add("ReferenceCell")
x_coords = np.linspace(0, 1, point_num)
y_coords = np.linspace(0, 1, point_num)
coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
coords = np.column_stack((coords, rough_fn(coords[:, :1], coords[:, 1:])))


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

# # Next add fluid domain:
p1_s = gmsh.model.geo.addPoint(*coords[0, :2], 0.0, meshSize=mesh_size_ref)
p2_s = gmsh.model.geo.addPoint(*coords[tag(point_num-1, 0) - 1, :2], 
                               0.0, meshSize=mesh_size_ref)
p3_s = gmsh.model.geo.addPoint(*coords[tag(point_num-1, point_num-1) - 1, :2], 
                               0.0, meshSize=mesh_size_ref)
p4_s = gmsh.model.geo.addPoint(*coords[tag(0, point_num-1) - 1, :2], 
                               0.0, meshSize=mesh_size_ref)
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

translation_1 = [1, 0, 0, 0, 
                0, 1, 0, 1, 
                0, 0, 1, 0, 
                0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(2, [s5_s], [s3_s], translation_1)


translation_2 = [1, 0, 0, 1, 
                0, 1, 0, 0, 
                0, 0, 1, 0, 
                0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(2, [s4_s], [s6_s], translation_2)

gmsh.model.mesh.setSize([(0, 1), (0, 2), (0, 3), (0, 4)], mesh_size_ref)
gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size_ref)

### Mark subdomains for saving
gmsh.model.addPhysicalGroup(3, [1], 1, name="Dummy")

gmsh.model.mesh.generate(3)
gmsh.write('MeshCreation/3DMesh/ref_cell.msh')

if show_mesh and '-nopopup' not in sys.argv:
   gmsh.fltk.run()

# close gmsh
gmsh.finalize()
