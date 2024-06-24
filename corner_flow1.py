# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw3-venv-run
#     language: python
#     name: python3
# ---

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
from underworld3.utilities import mem_footprint

import os 
import numpy as np
import sympy

# %%
# dimensional quantities

plateSpeed = 2  # cm/year
boxLength = 1e6 # meters
boxHeight = 2e5 # meters
particleX = [3.13e4, 9.74e4, 2.02e5, 3.97e5]
particleZ = 4*[-2e5]
grainsPerParticle = 5000
timeStep  = 5e12 # seconds 
timeMax   = 4e15 # seconds   

res = 16
Vdeg = 2
Pdeg = int(Vdeg - 1)

# system parameters
viscosity   = 1.
Vx_right    = 2
Vy_bot      = 0.5

# general solver options
tol = 1e-6



# %%
# unit registry for ease in converting between units
u = uw.scaling.units

ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

KL = boxHeight * u.meter
Kt = timeStep * u.second
KM = 1 * u.kilogram

scaling_coefficients = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"] = KM
scaling_coefficients

# %%
ndim(plateSpeed * u.centimeter/u.year)

# %%
minX, maxX = 0, ndim(boxLength * u.meter)
minZ, maxZ = -ndim(boxHeight * u.meter), 0

print("min X, max X:", minX, maxX)
print("min Z, max Z:", minZ, maxZ)

# %% [markdown]
# ### Mesh and variable creation 

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords=(minX, minZ),
                                                maxCoords=(maxX, maxZ), 
                                                cellSize=ndim(boxHeight * u.meter) /res, 
                                                regular=False, 
                                                qdegree = 3
                                        )

# %%
# # visualise the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.jupyter_backend = "trame"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, -5.0]
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=1)
    #pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Pdeg)

# strain rate tensor variable 
str_rate = uw.discretisation.MeshVariable("E", meshbox, (meshbox.dim, meshbox.dim), degree = Vdeg)

# %%
x = meshbox.N.x
y = meshbox.N.y

Lmat = sympy.Matrix([[sympy.diff(v_soln.sym[0], x), sympy.diff(v_soln.sym[1], x)],
              [sympy.diff(v_soln.sym[0], y), sympy.diff(v_soln.sym[1], y)]])

strain_rate_tensor = 0.5*(Lmat + sympy.Transpose(Lmat))

# %%
# projections to solve for the gradients
strain_tensor_calc = uw.systems.Tensor_Projection(meshbox, tensor_Field = str_rate)
strain_tensor_calc.uw_function = strain_rate_tensor
strain_tensor_calc.smoothing = 0.0
strain_tensor_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
# ### Velocity field over the entire domain

# %%
prefactor = 2 * ndim(plateSpeed * u.centimeter/u.year)/np.pi

x = meshbox.N.x
z = meshbox.N.y

vx = prefactor * (sympy.atan2(x, -z) + x*z/((x + 1e-8)**2 + (z + 1e-8)**2))
vz = prefactor * (z**2)/((x + 1e-8)**2 + (z + 1e-8)**2)

v_field = vx*meshbox.N.i + vz*meshbox.N.j

display(vx)
display(vz)
display(v_field)

# %%
# calculate velocity field - which is stationary
with meshbox.access(v_soln):
    v_soln.data[:] = uw.function.evaluate(v_field, v_soln.coords)

strain_tensor_calc.solve() # calculate strain tensor - this depends on v field, so this is also stationary

# %%
# check if there are nans caused by divisions by zero
with meshbox.access(v_soln):
    print(v_soln.data[:, 0].min())
    print(v_soln.data[:, 0].max())
    print(v_soln.data[:, 1].min())
    print(v_soln.data[:, 1].max())

# %%
str_rate.sym # check symbolic form of strain rate

# %% [markdown]
# ### Add particles with grains

# %%
particles           = uw.swarm.Swarm(mesh = meshbox) 
particleID          = uw.swarm.SwarmVariable(name = "ID", swarm = particles, vtype = uw.VarType.SCALAR, proxy_continuous = False, proxy_degree = 1, dtype = int)
grains_str_tensor   = uw.swarm.SwarmVariable(name = "G", swarm = particles, vtype = uw.VarType.TENSOR, size = (2, 2))

# pack the positions
all_posx = [grainsPerParticle*[x] for x in particleX]
all_posx = np.array(all_posx).flatten()
all_posz = [grainsPerParticle*[z] for z in particleZ]
all_posz = np.array(all_posz).flatten()

# convert particle X and Y to numpy arrays
particleXZ = np.vstack([all_posx, all_posz]).T
particleXZ = particleXZ.copy(order = "C") # make array contiguous again. Transposing array makes it not C_CONTIGUOUS

# add particles
particles.add_particles_with_coordinates(ndim(particleXZ * u.meter))

# add IDs of the particles
with particles.access(particleID):
    for i, x in enumerate(particleX):

        idx = ndim(x * u.meter) == particles.particle_coordinates.data[:, 0] # get indices with same X coords
        particleID.data[idx] = i


# %%
# double check size of the particles 
with particles.access(particles.particle_coordinates):
    pos = particles.particle_coordinates.data
    print(pos)

    print(particles.data)


# %%
print(f"Non-dimensionalised timestep: {ndim(timeStep * u.second)}")
print(f"Non-dimensionalsied max time: {ndim(timeMax * u.second)}")

# %% [markdown]
# ### Let particles travel and calculate strain rate at the current positions

# %%
import matplotlib.pyplot as plt

time = 0

timeStep_nd = ndim(timeStep * u.second)

cntr = 0
while time < ndim(timeMax * u.second):

    print(time)

    # calculate the strain rate at the particles' current positions
    with particles.access(grains_str_tensor):
        
        # FIXME: uw.function.evaluate() produces a 2x2 tensor, but swarm variable has shape [:, 2, 2] 
        # grains_str_tensor.data[:] = uw.function.evaluate(str_rate.sym, particles.data)

        # temporary fix - assumes that strain tensor is symmetric
        dummy_str_rate = uw.function.evaluate(str_rate.sym, particles.particle_coordinates.data) # evaluate strain rate at current position
        grains_str_tensor.data[:, 0] = dummy_str_rate[:, 0, 0] 
        grains_str_tensor.data[:, 1] = dummy_str_rate[:, 0, 1]
        grains_str_tensor.data[:, 2] = dummy_str_rate[:, 1, 0]
        grains_str_tensor.data[:, 3] = dummy_str_rate[:, 1, 1]  

    #####
    # TODO: do NEXT steps in here
    #####
    
    # plot every 100 timesteps
    if cntr % 100 == 0:
        fig, ax = plt.subplots(dpi = 150)
        
        with particles.access(particles.particle_coordinates):
            pos = particles.particle_coordinates.data
        
        ax.scatter(pos[:, 0], pos[:, 1], s = 10)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_xlim([minX, maxX])
        ax.set_ylim([minZ, maxZ])
        ax.set_aspect("equal")
        ax.set_title(f"Timestep: {cntr}")

    # update location of particles
    particles.advection(v_soln.sym, timeStep_nd, order=2, corrector=False, evalf=False)

    time += timeStep_nd

    cntr += 1



# %% [markdown]
# ### Visualize the velocity field

# %%
filename = "cornerflow_test"
if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh =  vis.mesh_to_pv_mesh(meshbox)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)/333
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)

    # point sources at cell centres
    cpoints = np.zeros((meshbox._centroids[::4, 0].shape[0], 3))
    cpoints[:, 0] = meshbox._centroids[::4, 0]
    cpoints[:, 1] = meshbox._centroids[::4, 1]
    cpoint_cloud = pv.PolyData(cpoints)

    pvstream = pvmesh.streamlines_from_source(
        cpoint_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="forward",
        compute_vorticity=False,
        max_steps=25,
        surface_streamlines=True,
    )

    points = vis.meshVariable_to_pv_cloud(p_soln)
    points.point_data["P"] = vis.scalar_fn_to_pv_points(points, p_soln.sym)
    point_cloud = pv.PolyData(points)

    ## PLOTTING
    pl = pv.Plotter(window_size=(1000, 750))
    
    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Gray",
        show_edges=True,
        scalars="V",
        use_transparency=False,
        opacity=0.75,
    )

    # pl.add_points(
    #     point_cloud,
    #     cmap="coolwarm",
    #     render_points_as_spheres=False,
    #     point_size=10,
    #     opacity=0.5,
    # )

    pl.add_mesh(pvstream, opacity=0.4)

    #pl.remove_scalar_bar("P")
    #pl.remove_scalar_bar("V")

    # pl.screenshot(
    #     filename="{}.png".format(filename),
    #     window_size=(1280, 1280),
    #     return_img=False,
    # )
    pl.show()

    pvmesh.clear_data()
    pvmesh.clear_point_data()
