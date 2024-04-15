from dolfin import *
import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, bmat
from scipy import sparse
from scipy.spatial import cKDTree
import time 
from petsc4py import PETSc

save_name = ""

### Effective Parameters (depend on geometry)
cond_scale = 0.64
u_bc_eff = 0.306 
u_bar = 0.5 # depends on the boundary condition!
K = 0.017  

len_gamma = 1.65   
cell_size = 0.84
### Problem parameters
L, H = 1, 1
res = 32
distance_tol = 1.e-5 # some distance to map the different meshes 
                     # (has to be smaller than mesh size)
## Wheel (g = grinding)
kappa_g = 0.05
c_g = 2.0
rho_g = 2.0

## Fluid 
kappa_f = 10.0 * cond_scale
c_f = 2.0
rho_f = 2.0 

mu_scale = 1.0
u_bc_speed = 1.0

psi_eff = u_bc_eff * u_bc_speed

theta_cool = 10.0 # boundary condition temperatur

## Interface interaction
alpha = 5.0 * len_gamma
heat_production = 100.0 * len_gamma
interface_marker = 2

## Time 
T_int = [0, 5]
dt = 0.05
t_n = T_int[0]

#%%
### Mesh loading
fluid_mesh = IntervalMesh(res, 0, L)

dofs_f = len(fluid_mesh.coordinates())

wheel_mesh = RectangleMesh(Point(0, 0), Point(L, H), res, res)

dofs_g = len(wheel_mesh.coordinates())

facet_markers_solid = MeshFunction("size_t", wheel_mesh, 1)
## Fluid and solid interface
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        x1_b = near(x[1], 0)
        return on_boundary and x1_b 

Interface().mark(facet_markers_solid, interface_marker)

print("Size of domains")
print("Wheel dofs", dofs_g)
print("Fluid dofs", dofs_f)
#%%
### Grinding wheel Problem:
V_g = FunctionSpace(wheel_mesh, "CG", 1)
theta_g_old = Function(V_g)
theta_g_old.assign(Constant(theta_cool))

dx_g = Measure("dx", wheel_mesh)
ds_g = Measure("ds", wheel_mesh, subdomain_data=facet_markers_solid)


u_g = TrialFunction(V_g)
phi_g = TestFunction(V_g)

a_g = inner(kappa_g * grad(u_g), grad(phi_g))
a_g += c_g * rho_g/dt * inner(u_g, phi_g)
a_g *= dx_g

a_g += alpha * inner(u_g, phi_g) * ds_g(interface_marker)

f_g = inner(Constant(heat_production), phi_g) * ds_g(interface_marker)
f_g += c_g * rho_g/dt * inner(theta_g_old, phi_g) * dx_g


A_g = PETScMatrix()
assemble(a_g, tensor=A_g)
bi, bj, bv = A_g.mat().getValuesCSR()
M_g = csr_matrix((bv, bj, bi))

#%%
### Fluid Problem (Here we only have to solve for the pressure)
V_f = FunctionSpace(fluid_mesh, "CG", 1)
P_fluid = FunctionSpace(fluid_mesh, "CG", 1)

fluid_mesh_marker = MeshFunction("size_t", fluid_mesh, 0)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > L - DOLFIN_EPS

RightBoundary().mark(fluid_mesh_marker, 1)

theta_f_old = Function(V_f)
theta_f_old.assign(Constant(theta_cool))

dx_f = Measure("dx", fluid_mesh)
ds_f = Measure("ds", fluid_mesh, subdomain_data=fluid_mesh_marker)

p_stokes = Function(P_fluid)

## Temperature
u_f = TrialFunction(V_f)
phi_f = TestFunction(V_f)

a_f = inner(kappa_f * grad(u_f), grad(phi_f))
a_f += inner(c_f*rho_f * grad(u_f)[0], u_bar) * phi_f
a_f += cell_size*c_f*rho_f/dt * inner(u_f, phi_f)
a_f *= dx_f

a_f += alpha * inner(u_f, phi_f) * dx_f

f_f = cell_size*c_f*rho_f/dt * inner(theta_f_old, phi_f) * dx_f

bc = DirichletBC(V_f, Constant(theta_cool), 
                 "on_boundary && x[0] < DOLFIN_EPS")

## SUPG: (here left side constant in time -> ok to only compute once)
res_temp = c_f*rho_f*(cell_size*u_f - cell_size*theta_f_old 
                      + dt*inner(u_bar, grad(u_f)[0])) \
       - dt*kappa_f*div(grad(u_f))

h = 1.0/res
Pe = h * abs(u_bar) / (2 * kappa_f)
tau_temp = (math.cosh(Pe)/math.sinh(Pe) - 1/Pe) * h/(2.0*abs(u_bar)) 
supg_term = tau_temp*inner(u_bar, grad(phi_f)[0])*res_temp*dx_f
f_f += rhs(supg_term)
a_f += lhs(supg_term)

A_f = PETScMatrix()
assemble(a_f, tensor=A_f)
for boundary_c in [bc]:
    boundary_c.apply(A_f)

bi, bj, bv = A_f.mat().getValuesCSR()
M_f = csr_matrix((bv, bj, bi))

## Pressure:
p_fluid = TestFunction(P_fluid)
q_fluid = TrialFunction(P_fluid)

mu_fn = mu_scale * (theta_cool/theta_f_old)**2
a_fluid_flow = inner(K*grad(p_fluid)[0], q_fluid)

# SUPG
p_e = -h/2.0 * grad(q_fluid)[0]
a_fluid_flow += inner(K*grad(p_fluid)[0], p_e)

rhs_fluid_flow = inner(mu_fn*(u_bar - psi_eff), q_fluid + p_e) 

A_stokes = a_fluid_flow * dx_f - inner(K*p_fluid, q_fluid) * ds_f(1)
L_stokes = rhs_fluid_flow * dx_f + inner(mu_fn*(u_bar - psi_eff), h * q_fluid) * ds_f(1)

helper_bc = DirichletBC(P_fluid, Constant(0.0), "on_boundary and x[0] < DOLFIN_EPS")

# A = assemble(A_stokes)
# L = assemble(L_stokes)
# helper_bc.apply(A, L)
# print(A.array())

# print(L.get_local())

# exit()

#%%
### Build matrix for complete system and construct coupling
M_complete = csr_matrix(bmat([[M_g, None], [None, M_f]]))
M_coupling = lil_matrix(M_complete) # best format for coupling

## For coupling first find the mapping from fluid verticies to other mesh
## (Assume they have the same mesh/vertex structure)
mass_matrix_fluid = assemble(inner(u_f, phi_f) * dx_f)

## Match vertex coordinates
wheel_kdtree = cKDTree(wheel_mesh.coordinates())
fluid_mesh_vertex_2d = np.column_stack((fluid_mesh.coordinates(), 
                                        np.zeros((dofs_f, 1))))
distance_f_to_g, mapping_f_to_g = wheel_kdtree.query(fluid_mesh_vertex_2d, 
                                       distance_upper_bound=distance_tol)
connected_vertex = np.where(distance_f_to_g < distance_tol)[0]

v_to_dof_g = vertex_to_dof_map(V_g)
v_to_dof_f = vertex_to_dof_map(V_f)
dof_to_v_f = dof_to_vertex_map(V_f)

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
        k = dof_to_v_f[dof_f_k]
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
M_coupling = csr_matrix(M_coupling)
petsc_mat = PETSc.Mat().createAIJ(size=M_coupling.shape, 
                csr=(M_coupling.indptr, M_coupling.indices, M_coupling.data))

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
solver.setOperators(petsc_mat)
solver.setType(PETSc.KSP.Type.PREONLY) #PREONLY, GMRES
solver.getPC().setType(PETSc.PC.Type.LU)

file_g     = File("Results/2DResults/Eff" + save_name + "/theta_g.pvd")
file_f     = File("Results/2DResults/Eff" + save_name + "/theta_f.pvd")
file_press = File("Results/2DResults/Eff" + save_name + "/pressure_f.pvd")

file_g << (theta_g_old, t_n)
file_f << (theta_f_old, t_n)
file_press << (p_stokes, t_n)

save_idx = 0

while t_n < T_int[1]:
    t_n += dt
    print("Working on time step", t_n)
    
    ## Fluid problem
    print("Start solving stokes")
    start_time = time.time()
    solve(A_stokes==L_stokes, p_stokes, [helper_bc])
    p_stokes.vector().set_local(
        p_stokes.vector().get_local() - assemble(p_stokes * dx_f) / L
    )
    print("Solving is done, took", time.time()- start_time)
    
    ## Temperatur problem
    # Build rhs 
    F_f = assemble(f_f)
    bc.apply(F_f)
    current_rhs = np.concatenate([assemble(f_g), F_f])
    petsc_vec.array[:] = current_rhs

    # solve temperatur problem
    print("Start solving")
    start_time = time.time()
    solver.solve(petsc_vec, u_sol_petsc)
    print("Solving is done, took", time.time()- start_time)

    theta_g_old.vector().set_local(u_sol_petsc[:dofs_g])
    theta_f_old.vector().set_local(u_sol_petsc[dofs_g:])

    if save_idx % 4 == 0:
        file_g << (theta_g_old, t_n)
        file_f << (theta_f_old, t_n)
        file_press << (p_stokes, t_n)
        save_idx = 0
    save_idx += 1