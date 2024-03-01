from dolfin import *

bc_movement_dir = Constant((1, 0, 0)) # normalized direction of flat boundary

######################
### Domain and measure
domain_mesh = Mesh()
with XDMFFile("MeshCreation/3DMesh/ref_cell.xdmf") as infile:
   infile.read(domain_mesh)

domain_mesh = UnitCubeMesh(8, 8, 8)

bc_marker = MeshFunction("size_t", domain_mesh, 2)

dx = Measure("dx", domain_mesh)
ds = Measure("ds", domain_mesh)

def bottom_boundary(x, on_boundary):
    return on_boundary and x[2] < DOLFIN_EPS

def top_boundary(x, on_boundary):
    at_1 = x[2] > 1 - DOLFIN_EPS 
    not_x0_boundary = (x[0] > DOLFIN_EPS and x[0] < 1 - DOLFIN_EPS)
    not_x2_boundary = (x[1] > DOLFIN_EPS and x[1] < 1 - DOLFIN_EPS)
    inside = x[2] > DOLFIN_EPS and not_x0_boundary and not_x2_boundary
    return on_boundary and (at_1 or inside) 

def both_boundaries(x, on_boundary):
    return bottom_boundary(x, on_boundary) or top_boundary(x, on_boundary)


print("cell volume:", assemble(1 * dx))
print("interface length (not always correct):", assemble(1 * ds)-5)


class MarkHelper(SubDomain): # This will later on become 1 free surface
    def __init__(self, mark_fn):
        SubDomain.__init__(self)
        self.fn = mark_fn

    def inside(self, x, on_boundary):
        return self.fn(x, on_boundary)

bc_marker.set_all(0)
MarkHelper(top_boundary).mark(bc_marker, 1)
MarkHelper(bottom_boundary).mark(bc_marker, 2)

filev = File("Results/3DResults/cell_problem/bc_marker.pvd")
filev << bc_marker

for i in range(2):
    print("working on direction:", i)
    if i == 0:
        e_j = Constant((1.0, 0.0, 0.0))
    else:
        e_j = Constant((0.0, 1.0, 0.0))

    ######################
    ### First temperature conductivity
    ### Functionspace stuff
    class PeriodicBC(SubDomain):

        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two slave edges
            return bool((near(x[0], 0) or near(x[1], 0)) and 
                        (not ((near(x[0], 1) and near(x[1], 0)) or 
                              (near(x[0], 0) and near(x[1], 1)))) 
                         and on_boundary)

        def map(self, x, y):
            if near(x[0], 1) and near(x[1], 1):
                y[0] = x[0] - 1
                y[1] = x[1] - 1 
                y[2] = x[2]
            elif near(x[0], 1):
                y[0] = x[0] - 1
                y[1] = x[1]
                y[2] = x[2]
            elif near(x[1], 1):
                y[0] = x[0]
                y[1] = x[1] - 1 
                y[2] = x[2]
            else:
                y[0] = -1000
                y[1] = -1000
                y[2] = -1000

    V = FiniteElement('CG', domain_mesh.ufl_cell(), 1)
    R = FiniteElement('R', domain_mesh.ufl_cell(), 0)
    mixed = MixedElement([V, R])
    W = FunctionSpace(domain_mesh, mixed, constrained_domain=PeriodicBC())

    u, lamb = TrialFunctions(W)
    v, mu = TestFunctions(W)

    a = inner(grad(u), grad(v)) + inner(lamb, v) + inner(u, mu)
    a *= dx

    f = -inner(e_j, grad(v)) * dx
    w = Function(W)

    solve(a==f, w, solver_parameters={'linear_solver' : 'mumps'})

    u0, _ = w.split()

    diffusion_scale_x = assemble(inner((grad(u0) + e_j), Constant((1, 0, 0)))*dx)
    print(diffusion_scale_x)
    diffusion_scale_y = assemble(inner((grad(u0) + e_j), Constant((0, 1, 0)))*dx)
    print(diffusion_scale_y)
    diffusion_scale_z = assemble(inner((grad(u0) + e_j), Constant((0, 0, 1)))*dx)
    print(diffusion_scale_z)
    # print("Other computaion")
    # diffusion_scale = assemble(inner((grad(u0) + e_j), (grad(u0) + e_j))*dx)
    # print(diffusion_scale)

    filev = File("Results/3DResults/cell_problem/temp_dir_" +  str(i) + ".pvd")
    filev << u0

    ######################
    ### Next permeability
    V = VectorElement('CG', domain_mesh.ufl_cell(), 2)
    P = FiniteElement('CG', domain_mesh.ufl_cell(), 1)
    Lamb = FiniteElement('R', domain_mesh.ufl_cell(), 0)
    mixed = MixedElement([V, P, Lamb])
    W_stokes = FunctionSpace(domain_mesh, mixed, constrained_domain=PeriodicBC())

    u, p, lamb = TrialFunctions(W_stokes)
    v, q, chi = TestFunctions(W_stokes)

    momentum = inner(grad(u), grad(v)) + inner(div(v), p)
    mass = div(u)*q

    rhs_fluid = inner(e_j, v) 

    A_stokes = (momentum + mass) * dx
    L_stokes = rhs_fluid * dx

    #A_stokes += 1.e-7*p*q*dx # trick to zero average from fenics forum
    A_stokes += p * chi * dx + q * lamb * dx

    noslip_bc = DirichletBC(W_stokes.sub(0), Constant((0, 0, 0)), both_boundaries)

    w = Function(W_stokes)
    solve(A_stokes==L_stokes, w, [noslip_bc]) 
        #  solver_parameters={'linear_solver' : 'mumps'})

    u0, _, _ = w.split()

    permeability_x = assemble(inner(u0, Constant((1, 0, 0)))*dx)
    print(permeability_x)
    permeability_y = assemble(inner(u0, Constant((0, 1, 0)))*dx)
    print(permeability_y)
    permeability_z = assemble(inner(u0, Constant((0, 0, 1)))*dx)
    print(permeability_z)
    filev = File("Results/3DResults/cell_problem/u_perm_dir_" +  str(i) + ".pvd")
    filev << u0

##########################
### Flow given by boundary movement
noslip_bc = DirichletBC(W_stokes.sub(0), Constant((0, 0, 0)), top_boundary)
speed_bc  = DirichletBC(W_stokes.sub(0), bc_movement_dir, bottom_boundary)

w = Function(W_stokes)
solve(A_stokes==L_stokes, w, [noslip_bc, speed_bc])

u0, _, _ = w.split()

speed_x = assemble(inner(u0, Constant((1, 0, 0)))*dx)
print(speed_x)
speed_y = assemble(inner(u0, Constant((0, 1, 0)))*dx)
print(speed_y)
speed_z = assemble(inner(u0, Constant((0, 0, 1)))*dx)
print(speed_z)

filev = File("Results/3DResults/cell_problem/u_bc_movement.pvd")
filev << u0