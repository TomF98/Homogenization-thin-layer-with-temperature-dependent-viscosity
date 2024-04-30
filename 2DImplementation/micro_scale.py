from dolfin import *
import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, bmat
from scipy import sparse
from scipy.spatial import cKDTree
import time 
from petsc4py import PETSc


eps_list = [0.05]
gamma_0_list = [0.5]
name_list = ["rect"]

for name in name_list:
    for gamma_0 in gamma_0_list:
        for eps in eps_list:
            ### Problem parameters
            #eps = 0.05
            #name = "sin"
            inflow_typ = "heat_scaled_linear"
            #gamma_0 = 0.5

            L, H = 1, 1

            distance_tol = 1.e-5 # some distance to map the different meshes (smaller H)
            ## Wheel (g = grinding)
            kappa_g = 0.5
            c_g = 1.0
            rho_g = 1.0

            ## Fluid 
            kappa_f = 0.1
            c_f = 1.0
            rho_f = 1.0 

            mu_scale_A = 0.2
            mu_scale_B = 3.0
            mu_scale_C = 0.6

            u_bc_speed = 1.0

            theta_cool = 0.0 # boundary condition temperatur

            ## Interface interaction
            alpha = 1.0
            interface_marker = 2
            #heat_production = Expression("(1-x[1]/e) / (1 - g)", degree=1, e=eps, g=gamma_0)
            
            ## Time 
            T_int = [0, 5]
            dt = 0.05
            t_n = T_int[0]

            #%%
            ### Mesh loading
            fluid_mesh = Mesh()
            with XDMFFile("MeshCreation/2DMesh/XDMF/"+ str(name) +"_fluid_domain_gamma0_" 
                        + str(gamma_0) + "_eps_" + str(eps) +'.xdmf') as infile:
                infile.read(fluid_mesh)

            dofs_f = len(fluid_mesh.coordinates())

            wheel_mesh = Mesh()
            with XDMFFile("MeshCreation/2DMesh/XDMF/"+ str(name) +"_solid_domain_gamma0_" 
                        + str(gamma_0) + "_eps_" + str(eps) +'.xdmf') as infile:
                infile.read(wheel_mesh)

            dofs_g = len(wheel_mesh.coordinates())

            facet_markers_fluid = MeshFunction("size_t", fluid_mesh, 1)
            facet_markers_solid = MeshFunction("size_t", wheel_mesh, 1)
            ## Fluid and solid interface
            class Interface(SubDomain):
                def inside(self, x, on_boundary):
                    x1_b = near(x[1], H) or near(x[1], 0)
                    x0_b = near(x[0], L) or near(x[0], 0)
                    return on_boundary and not x0_b and not x1_b 

            Interface().mark(facet_markers_fluid, interface_marker)
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

            print("interface length", assemble(1 * ds_g(interface_marker)))
            heat_production = Constant(1.0/assemble(1 * ds_g(interface_marker)))
            print("heat source", assemble(heat_production * ds_g(interface_marker)))
            u_g = TrialFunction(V_g)
            phi_g = TestFunction(V_g)

            heat_source = interpolate(heat_production, V_g)

            a_g = inner(kappa_g * grad(u_g), grad(phi_g))
            a_g += c_g * rho_g/dt * inner(u_g, phi_g)
            a_g *= dx_g

            a_g += alpha * inner(u_g, phi_g) * ds_g(interface_marker)

            f_g = inner(heat_source, phi_g) * ds_g(interface_marker)
            f_g += c_g * rho_g/dt * inner(theta_g_old, phi_g) * dx_g

            A_g = PETScMatrix()
            assemble(a_g, tensor=A_g)
            bi, bj, bv = A_g.mat().getValuesCSR()
            M_g = csr_matrix((bv, bj, bi))

            #%%
            ### Fluid Problem:
            V_f = FunctionSpace(fluid_mesh, "CG", 1)

            V_fluid = VectorElement("CG", fluid_mesh.ufl_cell(), 2)
            P_fluid = FiniteElement("CG", fluid_mesh.ufl_cell(), 1)
            W_fluid_elem = MixedElement([V_fluid, P_fluid])
            W_fluid = FunctionSpace(fluid_mesh, W_fluid_elem) 

            theta_f_old = Function(V_f)
            theta_f_old.assign(Constant(theta_cool))

            dx_f = Measure("dx", fluid_mesh)
            ds_f = Measure("ds", fluid_mesh, subdomain_data=facet_markers_fluid)

            # print(name + " eps: " + str(eps) + " gamma: " + str(gamma_0))
            # print(assemble(1 * ds_f(interface_marker)), assemble(1 * dx_f))
            # break

            w_stokes = Function(W_fluid)
            v_stokes, p_stokes = split(w_stokes)
            ## Temperature
            u_f = TrialFunction(V_f)
            phi_f = TestFunction(V_f)

            a_f = inner(kappa_f/eps * grad(u_f), grad(phi_f))
            a_f += inner(c_f*rho_f/eps * grad(u_f), v_stokes) * phi_f
            a_f += 1/eps * c_f*rho_f/dt * inner(u_f, phi_f)
            a_f *= dx_f

            a_f += alpha * inner(u_f, phi_f) * ds_f(interface_marker)

            f_f = 1/eps * c_f*rho_f/dt * inner(theta_f_old, phi_f) * dx_f

            ## SUPG:
            res = c_f*rho_f*(u_f - theta_f_old + dt*inner(v_stokes, grad(u_f))) \
                - dt*kappa_f*div(grad(u_f))

            vnorm = sqrt(inner(v_stokes, v_stokes))
            h = CellDiameter(fluid_mesh)
            tau = 10*h/(2.0*vnorm) 
            supg_term = tau*inner(v_stokes, grad(phi_f))*res*dx_f
            f_f += rhs(supg_term)
            a_f += lhs(supg_term)

            bc = DirichletBC(V_f, Constant(theta_cool), 
                            "on_boundary && x[0] <= DOLFIN_EPS")

            A_f = PETScMatrix()
            assemble(a_f, tensor=A_f)
            for boundary_c in [bc]:
                boundary_c.apply(A_f)

            bi, bj, bv = A_f.mat().getValuesCSR()
            M_f = csr_matrix((bv, bj, bi))

            ## Fluid:
            u_fluid, p_fluid = TestFunctions(W_fluid)
            psi_fluid, q_fluid = TrialFunctions(W_fluid)

            mu_fn = mu_scale_A * exp(mu_scale_B/(theta_f_old + mu_scale_C))
            momentum = inner(mu_fn * grad(u_fluid), grad(psi_fluid)) + inner(div(psi_fluid), p_fluid)
            mass = div(u_fluid)*q_fluid

            rhs_fluid = inner(Constant(0.0), q_fluid) 

            A_stokes = (momentum + mass) * dx_f
            L_stokes = rhs_fluid * dx_f

            A_stokes += 1.e-7*p_fluid*q_fluid*dx_f # trick to zero average from fenics forum

            ## Boundary conditions
            class FlowBC(UserExpression):
                def eval(self, values, x):
                    values[1] = 0
                    if x[1] < DOLFIN_EPS:
                        values[0] = u_bc_speed
                    elif (x[0] < DOLFIN_EPS or x[0] > L - DOLFIN_EPS) and x[1] <= eps + DOLFIN_EPS:
                        values[0] = u_bc_speed * (1 - (x[1]/(eps)))
                    else:
                        values[0] = 0

                def value_shape(self):
                    return (2,)

            flow_bc_function = interpolate(FlowBC(), W_fluid.sub(0).collapse())
            stokes_bc = DirichletBC(W_fluid.sub(0), flow_bc_function, "on_boundary")

            #%%
            ### Build matrix for complete system and construct coupling
            M_coupling = lil_matrix((dofs_g + dofs_f, dofs_g + dofs_f)) # best format for coupling

            ## For coupling first find the mapping from fluid verticies to other mesh
            ## (Assume they have the same mesh/vertex structure)
            mass_matrix_fluid = assemble(inner(u_f, phi_f) * ds_f(interface_marker))

            ## Match vertex coordinates
            wheel_kdtree = cKDTree(wheel_mesh.coordinates())
            distance_f_to_g, mapping_f_to_g = wheel_kdtree.query(fluid_mesh.coordinates(), 
                                                distance_upper_bound=distance_tol)
            connected_vertex = np.where(distance_f_to_g < distance_tol)[0]

            v_to_dof_g = vertex_to_dof_map(V_g)
            v_to_dof_f = vertex_to_dof_map(V_f)
            dof_to_v_f = dof_to_vertex_map(V_f)

            for idx_f in connected_vertex:
                dof_f = v_to_dof_f[idx_f]

                coupling_dofs, mass_values = mass_matrix_fluid.getrow(dof_f)
                coupling_dofs = coupling_dofs.astype(np.int32)
                idx_g = v_to_dof_g[mapping_f_to_g[idx_f]]

                ## Set coupling in matrix:
                counter = 0
                for dof_f_k in coupling_dofs:
                    k = dof_to_v_f[dof_f_k]
                    if distance_f_to_g[k] < distance_tol:
                        # check if we are at the left boundary:
                        dirichlet_point = fluid_mesh.coordinates()[k][0] < DOLFIN_EPS
                        if not dirichlet_point:
                            M_coupling[dofs_g + dof_f_k, idx_g] -= alpha * mass_values[counter]
                        
                        M_coupling[idx_g, dofs_g + dof_f_k] -= alpha * mass_values[counter]

                    counter += 1

            # Show matrix        
            # import matplotlib.pylab as plt
            # plt.spy(M_coupling)
            # plt.show()

            ### Finish up coupling matrix and transform back to PETSc
            M_coupling = csr_matrix(M_coupling)

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
            solver.setType(PETSc.KSP.Type.PREONLY) #PREONLY, GMRES
            solver.getPC().setType(PETSc.PC.Type.LU)


            save_name = name + "_eps_" + str(eps) + "_gamma_" + str(gamma_0) + "_inflow_" + inflow_typ
            file_g     = File("Results/2DResults/Micro/" + save_name + "/theta_g.pvd")
            file_f     = File("Results/2DResults/Micro/" + save_name + "/theta_f.pvd")
            file_flow  = File("Results/2DResults/Micro/" + save_name + "/fluid_f.pvd")
            file_press = File("Results/2DResults/Micro/" + save_name + "/pressure_f.pvd")

            file_g << (theta_g_old, t_n)
            file_f << (theta_f_old, t_n)
            v_stokes_save, p_stokes_save = w_stokes.split()
            file_flow << (v_stokes_save, t_n)
            file_press << (p_stokes_save, t_n)

            save_idx = 0

            while t_n < T_int[1] - dt/4.0:
                t_n += dt
                print("Working on time step", t_n)
                
                ## Fluid problem
                print("Start solving stokes")
                start_time = time.time()
                solve(A_stokes==L_stokes, w_stokes, [stokes_bc], 
                    solver_parameters={'linear_solver' : 'mumps'})
                print("Solving is done, took", time.time()- start_time)
                ## Temperatur problem
                # Update convection part:
                A_f = PETScMatrix()
                assemble(a_f, tensor=A_f)
                for boundary_c in [bc]:
                    boundary_c.apply(A_f)

                bi, bj, bv = A_f.mat().getValuesCSR()
                M_f = csr_matrix((bv, bj, bi))

                M_complete = csr_matrix(bmat([[M_g, None], [None, M_f]])) + M_coupling
                petsc_mat = PETSc.Mat().createAIJ(size=M_complete.shape, 
                            csr=(M_complete.indptr, M_complete.indices, M_complete.data))
                solver.setOperators(petsc_mat)

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
                    file_flow << (v_stokes_save, t_n)
                    file_press << (p_stokes_save, t_n)
                    save_idx = 0
                save_idx += 1

            # save last step:
            if save_idx > 1:
                file_g << (theta_g_old, t_n)
                file_f << (theta_f_old, t_n)
                file_flow << (v_stokes_save, t_n)
                file_press << (p_stokes_save, t_n)