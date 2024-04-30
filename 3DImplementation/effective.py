from dolfin import *
import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, bmat
from scipy import sparse
from scipy.spatial import cKDTree
import time 
from petsc4py import PETSc

save_name = "temperature_viscosity"

### Effective Parameters (depend on geometry)
cond_scale = Constant(((0.4925, 0), (0, 0.635))) #Constant(((0.1, 0), (0, 0.5)))

K = Constant(((0.0046, 0.0), (0.0, 0.005))) #Constant(((0.1, 0.01), (-0.5, 0.9)))

u_bc_eff = Constant((0.134, 0))

u_bar = Constant((0.5, 0))

len_gamma = 1.025  
cell_size = 0.96
### Problem parameters
L, B, H = 1, 1, 1
res = 32
distance_tol = 1.e-5 # some distance to map the different meshes 
                     # (has to be smaller than mesh size)
## Wheel (g = grinding)
kappa_g = 0.5
c_g = 1.0
rho_g = 1.0

class HeatSoruce(UserExpression):

    def eval(self, values, x):
        if x[1] > L/2.0 and abs(x[0] - 0.5) < 0.1:
            values[0] = 2.0
        else:
            values[0] = 0.0

heat_production = HeatSoruce()

## Fluid 
kappa_f = 0.005 * cond_scale
c_f = 1.0
rho_f = 1.0 

mu_scale_A = 0.2
mu_scale_B = 3.0
mu_scale_C = 0.4

u_bc_speed = 1.0

psi_eff = u_bc_eff * u_bc_speed

theta_cool = 0.0 # boundary condition temperatur

## Interface interaction
alpha = 1.0 * len_gamma
interface_marker = 2
boundary_in_marker = 1
boundary_out_marker = 2

## Time 
T_int = [0, 5]
dt = 0.05
t_n = T_int[0]

#%%
### Mesh loading
fluid_mesh = RectangleMesh(Point(0, 0), Point(L, B), res, res)

dofs_f = len(fluid_mesh.coordinates())

wheel_mesh = BoxMesh(Point(0, 0, 0), Point(L, B, H), res, res, res)

dofs_g = len(wheel_mesh.coordinates())

facet_markers_solid = MeshFunction("size_t", wheel_mesh, 2)
## Fluid and solid interface
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        x1_b = near(x[2], 0)
        return on_boundary and x1_b 

Interface().mark(facet_markers_solid, interface_marker)

print("Size of domains")
print("Wheel dofs", dofs_g)
print("Fluid dofs", dofs_f)

file_domain = File("Results/3DResults/Eff/" + save_name + "/box_mesh.pvd")
file_domain << facet_markers_solid
file_domain = File("Results/3DResults/Eff/" + save_name + "/fluid_mesh.pvd")
file_domain << fluid_mesh

#%%
### Grinding wheel Problem:
V_g = FunctionSpace(wheel_mesh, "CG", 1)
theta_g_old = Function(V_g)
theta_g_old.assign(Constant(theta_cool))

dx_g = Measure("dx", wheel_mesh)
ds_g = Measure("ds", wheel_mesh, subdomain_data=facet_markers_solid)

heat_production = interpolate(heat_production, V_g)
file_domain = File("Results/3DResults/Eff/" + save_name + "/heat_source.pvd")
file_domain << heat_production

u_g = TrialFunction(V_g)
phi_g = TestFunction(V_g)

a_g = inner(kappa_g * grad(u_g), grad(phi_g))
a_g += c_g * rho_g/dt * inner(u_g, phi_g)
a_g *= dx_g

a_g += alpha * inner(u_g, phi_g) * ds_g(interface_marker)

f_g = inner(heat_production, phi_g) * ds_g(interface_marker)
f_g += c_g * rho_g/dt * inner(theta_g_old, phi_g) * dx_g


A_g = PETScMatrix()
assemble(a_g, tensor=A_g)
bi, bj, bv = A_g.mat().getValuesCSR()
M_g = csr_matrix((bv, bj, bi))

#%%
### Fluid Problem 
Theta_f = FunctionSpace(fluid_mesh, "CG", 1)
V_f = VectorFunctionSpace(fluid_mesh, "CG", 1)

n_sigma = FacetNormal(fluid_mesh)

fluid_mesh_marker = MeshFunction("size_t", fluid_mesh, 0)

class InBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS

class OutBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > DOLFIN_EPS

InBoundary().mark(fluid_mesh_marker, boundary_in_marker)
OutBoundary().mark(fluid_mesh_marker, boundary_out_marker)

theta_f_old = Function(Theta_f)
theta_f_old.assign(Constant(theta_cool))

u_eff = Function(V_f)
p_stokes = Function(Theta_f)

dx_f = Measure("dx", fluid_mesh)
ds_f = Measure("ds", fluid_mesh, subdomain_data=fluid_mesh_marker)

## Temperature
temp_f = TrialFunction(Theta_f)
phi_f = TestFunction(Theta_f)

a_f = inner(kappa_f * grad(temp_f), grad(phi_f))
a_f += inner(c_f*rho_f * grad(temp_f), u_eff) * phi_f
a_f += cell_size*c_f*rho_f/dt * inner(temp_f, phi_f)
a_f *= dx_f

a_f += alpha * inner(temp_f, phi_f) * dx_f

f_f = cell_size*c_f*rho_f/dt * inner(theta_f_old, phi_f) * dx_f

bc = DirichletBC(Theta_f, Constant(theta_cool), "on_boundary && x[0] < DOLFIN_EPS")

## SUPG: 
res_temp = c_f*rho_f*(cell_size*temp_f - cell_size*theta_f_old + dt*inner(u_eff, grad(temp_f))) \
                - dt*div(kappa_f*grad(temp_f))

h = 1.0/res
tau_temp = 1.0 * h/(2.0*sqrt(inner(u_eff, u_eff)) + 0.001) # add small value since at the start 0 
supg_term = tau_temp*inner(u_eff, grad(phi_f))*res_temp*dx_f
supg_f = rhs(supg_term)
supg_a = lhs(supg_term)

A_f = PETScMatrix()
assemble(a_f + supg_a, tensor=A_f)
for boundary_c in [bc]:
    boundary_c.apply(A_f)

bi, bj, bv = A_f.mat().getValuesCSR()
M_f = csr_matrix((bv, bj, bi))

## Pressure:
p_fluid = TrialFunction(Theta_f)
q_fluid = TestFunction(Theta_f)

mu_fn = mu_scale_A * exp(mu_scale_B/(theta_f_old + mu_scale_C))

a_pressure = inner(K/mu_fn * grad(p_fluid), grad(q_fluid)) * dx_f
f_pressure = -inner(u_bar - psi_eff, n_sigma) * q_fluid * ds_f 

A_stokes = a_pressure
L_stokes = f_pressure

helper_bc = DirichletBC(Theta_f, Constant(0.0), "on_boundary and x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS",
                        method="pointwise")

# A = assemble(A_stokes)
# L = assemble(L_stokes)
# helper_bc.apply(A, L)
# print(A.array())

# print(L.get_local())

#%%
### Build matrix for complete system and construct coupling
M_complete = csr_matrix(bmat([[M_g, None], [None, M_f]]))
M_coupling = lil_matrix(M_complete) # best format for coupling

## For coupling first find the mapping from fluid verticies to other mesh
## (Assume they have the same mesh/vertex structure)
mass_matrix_fluid = assemble(inner(temp_f, phi_f) * dx_f)

## Match vertex coordinates
wheel_kdtree = cKDTree(wheel_mesh.coordinates())
fluid_mesh_vertex_3D = np.column_stack((fluid_mesh.coordinates(), 
                                        np.zeros((dofs_f, 1))))
distance_f_to_g, mapping_f_to_g = wheel_kdtree.query(fluid_mesh_vertex_3D, 
                                       distance_upper_bound=distance_tol)
connected_vertex = np.where(distance_f_to_g < distance_tol)[0]

v_to_dof_g = vertex_to_dof_map(V_g)
v_to_dof_f = vertex_to_dof_map(Theta_f)
dof_to_Theta_f = dof_to_vertex_map(Theta_f)

for idx_f in connected_vertex:
    dof_f = v_to_dof_f[idx_f]

    coupling_dofs, mass_values = mass_matrix_fluid.getrow(dof_f)
    coupling_dofs = coupling_dofs.astype(np.int32)
    #print(coupling_dofs, mass_values)
    # check if we are at the left boundary:
    idx_g = v_to_dof_g[mapping_f_to_g[idx_f]]
    
    ## Set coupling in matrix:
    counter = 0
    for dof_f_k in coupling_dofs:
        k = dof_to_Theta_f[dof_f_k]
        if distance_f_to_g[k] < distance_tol:
            dirichlet_point = fluid_mesh.coordinates()[k][0] < DOLFIN_EPS
            if not dirichlet_point:
                M_coupling[dofs_g + dof_f_k, idx_g] -= alpha * mass_values[counter]
            
            M_coupling[idx_g, dofs_g + dof_f_k] -= alpha * mass_values[counter]

        counter += 1

### Show matrix        
# import matplotlib.pylab as plt
# plt.spy(M_coupling)
# plt.show()

### Finish up coupling matrix and transform back to PETSc
### Create vectors for writting the solution and rhs
petsc_vec = PETSc.Vec()
petsc_vec.create(PETSc.COMM_WORLD)
petsc_vec.setSizes(dofs_g + dofs_f)
petsc_vec.setUp()

u_sol_petsc = PETSc.Vec()
u_sol_petsc.create(PETSc.COMM_WORLD)
u_sol_petsc.setSizes(dofs_g + dofs_f)
u_sol_petsc.setUp()

solver = PETSc.KSP().create()
solver.setType(PETSc.KSP.Type.GMRES) #PREONLY, GMRES
#solver.getPC().setType(PETSc.PC.Type.LU)

file_g     = File("Results/3DResults/Eff/" + save_name + "/theta_g.pvd")
file_f     = File("Results/3DResults/Eff/" + save_name + "/theta_f.pvd")
file_press = File("Results/3DResults/Eff/" + save_name + "/pressure_f.pvd")
file_u = File("Results/3DResults/Eff/" + save_name + "/u_f.pvd")

file_g << (theta_g_old, t_n)
file_f << (theta_f_old, t_n)
file_press << (p_stokes, t_n)
file_u << (u_eff, t_n)

save_idx = 0

while t_n < T_int[1]:
    t_n += dt
    print("Working on time step", t_n)

    ## Fluid problem
    print("Start solving stokes")
    start_time = time.time()
    solve(A_stokes==L_stokes, p_stokes, [helper_bc])
    p_stokes.vector().set_local(
        p_stokes.vector().get_local() - assemble(p_stokes * dx_f) / (L*B)
    )
    print("Solving is done, took", time.time()- start_time)
    
    ## Compute effective u
    u_eff.assign(project(-K/mu_fn * grad(p_stokes) + u_bc_eff, V_f))

    ## Temperatur problem
    # Build rhs 
    A_f_current, F_f = assemble_system(a_f + supg_a, f_f + supg_f, bc, A_tensor=A_f)
    current_rhs = np.concatenate([assemble(f_g), F_f])
    petsc_vec.array[:] = current_rhs

    bi, bj, bv = A_f.mat().getValuesCSR()
    M_f = csr_matrix((bv, bj, bi))
    M_coupling[dofs_g:, dofs_g:] = M_f
    M_coupling_current = csr_matrix(M_coupling)
    petsc_mat = PETSc.Mat().createAIJ(size=M_coupling_current.shape, 
                    csr=(M_coupling_current.indptr,
                         M_coupling_current.indices, 
                         M_coupling_current.data))
    
    # Solve temperatur problem
    print("Start solving")
    start_time = time.time()
    solver.setOperators(petsc_mat)
    solver.solve(petsc_vec, u_sol_petsc)
    print("Solving is done, took", time.time()- start_time)

    theta_g_old.vector().set_local(u_sol_petsc[:dofs_g])
    theta_f_old.vector().set_local(u_sol_petsc[dofs_g:])

    if save_idx % 4 == 0:
        file_g << (theta_g_old, t_n)
        file_f << (theta_f_old, t_n)
        file_press << (p_stokes, t_n)
        file_u << (u_eff, t_n)
        save_idx = 0
    save_idx += 1