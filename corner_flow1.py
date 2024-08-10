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

# copied from PyDRex/tests/test_corner_flow_2d.py
import contextlib as cl
import functools as ft
import pathlib as pl
from time import perf_counter

import numpy as np
import pytest

from pydrex import core as _core
#from pydrex import diagnostics as _diagnostics
#from pydrex import geometry as _geo
#from pydrex import io as _io
#from pydrex import logger as _log
from pydrex import minerals as _minerals
#from pydrex import pathlines as _path
#from pydrex import stats as _stats
from pydrex import utils as _utils
#from pydrex import velocity as _velocity
#from pydrex import visualisation as _vis

# %%
4e15/5e12

# %%
# dimensional quantities

plateSpeed = 2  # cm/year
boxLength = 1e6 # meters
boxHeight = 2e5 # meters
particleX = [3.13e4, 9.74e4, 2.02e5, 3.97e5]
particleZ = 4*[-2e5]
grainsPerParticle = 5000
timeStep  = 2.5e12 # seconds 
timeMax   = 500*timeStep # seconds   
#timeMax   = 4e15 # seconds  

res = 32
Vdeg = 2
Pdeg = int(Vdeg - 1)

# system parameters
viscosity   = 1.
Vx_right    = 2
Vy_bot      = 0.5

# general solver options
tol = 1e-6

# pydrex settings
params = _core.DefaultParams().as_dict()
params["number_of_grains"] = grainsPerParticle



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
#str_rate = uw.discretisation.MeshVariable("E", meshbox, (meshbox.dim, meshbox.dim), degree = Vdeg)

# velocity gradient 
vel_grad = uw.discretisation.MeshVariable("Vg", meshbox, (meshbox.dim, meshbox.dim), degree = Vdeg)

# %%
x = meshbox.N.x
y = meshbox.N.y

Lmat = sympy.Matrix([[sympy.diff(v_soln.sym[0], x), sympy.diff(v_soln.sym[1], x)],
              [sympy.diff(v_soln.sym[0], y), sympy.diff(v_soln.sym[1], y)]])

# strain_rate_tensor = 0.5*(Lmat + sympy.Transpose(Lmat))

# %%
# projections to solve for the gradients
# strain_tensor_calc = uw.systems.Tensor_Projection(meshbox, tensor_Field = str_rate)
# strain_tensor_calc.uw_function = strain_rate_tensor
# strain_tensor_calc.smoothing = 0.0
# strain_tensor_calc.petsc_options.delValue("ksp_monitor")

vel_grad_calc = uw.systems.Tensor_Projection(meshbox, tensor_Field = vel_grad)
vel_grad_calc.uw_function = Lmat
vel_grad_calc.smoothing = 0.0
vel_grad_calc.petsc_options.delValue("ksp_monitor")

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

# strain_tensor_calc.solve() # calculate strain tensor - this depends on v field, so this is also stationary
vel_grad_calc.solve()

# %%
# check if there are nans caused by divisions by zero
with meshbox.access(v_soln):
    print(v_soln.data[:, 0].min())
    print(v_soln.data[:, 0].max())
    print(v_soln.data[:, 1].min())
    print(v_soln.data[:, 1].max())

# %%
# str_rate.sym # check symbolic form of strain rate
vel_grad.sym

# %% [markdown]
# ### Add particles with grains

# %%
particles           = uw.swarm.Swarm(mesh = meshbox) 
particleID          = uw.swarm.SwarmVariable(name = "ID", swarm = particles, vtype = uw.VarType.SCALAR, proxy_continuous = False, proxy_degree = 1, dtype = int)
grains_str_tensor   = uw.swarm.SwarmVariable(name = "G", swarm = particles, vtype = uw.VarType.TENSOR, size = (2, 2))

# pack the positions
all_posx = np.array(particleX)
all_posz = np.array(particleZ)

# convert particle X and Y to numpy arrays
particleXZ = np.vstack([all_posx, all_posz]).T
particleXZ = particleXZ.copy(order = "C") # make array contiguous again. Transposing array makes it not C_CONTIGUOUS

# add particles
particles.add_particles_with_coordinates(ndim(particleXZ * u.meter))

#list of pydrex mineral object
mineral_list = []

# add IDs of the particles
# currently not using the IDs
with particles.access(particleID):
    for i, x in enumerate(particleX):

        idx = ndim(x * u.meter) == particles.particle_coordinates.data[:, 0] # get indices with same X coords
        particleID.data[idx] = i

        mineral_list.append(_minerals.Mineral(
                                _core.MineralPhase.olivine,
                                _core.MineralFabric.olivine_A,
                                _core.DeformationRegime.matrix_dislocation,
                                n_grains = grainsPerParticle,
                                seed = 42,
                        ))

# %%
print(mineral_list[0].orientations[0].shape)

# %%
# double check size of the particles 
with particles.access(particles):
    pos = particles.particle_coordinates.data
    for p in pos:
        print(np.append(p, 0.))

    #print(particles.data)

# %%
print(f"Non-dimensionalised timestep: {ndim(timeStep * u.second)}")
print(f"Non-dimensionalsied max time: {ndim(timeMax * u.second)}")

# %%
# functions needed by pydrex
def get_velocity_gradient(t, x):
    '''
    calculates the velocity gradient at position x
    x - a np.array containing the position (2D position for now)
    '''
    L = np.zeros([3, 3])
    L[0:2, 0:2] = uw.function.evaluate(vel_grad.sym, x[0:2].reshape(1, 2)) # FIXME: doing it like this will be inefficient

    return L

# %%
get_velocity_gradient(np.nan, np.array([[0., 0.]]))

# %%
timestamps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

for t, time in enumerate(timestamps[:-1], start=1):
    
    print(t, time, timestamps[t])
        # strains[t] = strains[t - 1] + (
        #     _utils.strain_increment(timestamps[t] - time, velocity_gradients[t]) 
        # ) # current_t - prev_t, velocity_gradient of current_t 
        # _log.info(
        #     "final location = %s; step %d/%d (ε = %.2f)",
        #     final_location.ravel(),
        #     t,
        #     len(timestamps) - 1,
        #     strains[t],
        # )

        # deformation_gradient = mineral.update_orientations(
        #     params,
        #     deformation_gradient,
        #     get_velocity_gradient,
        #     pathline=(time, timestamps[t], get_position),
        # )

# %% [markdown]
# ### Let particles travel and calculate strain rate at the current positions

# %%
timeStep_nd = ndim(timeStep * u.second)
timeMax_nd = ndim(timeMax * u.second)

time_nd_arr = np.arange(0, timeMax_nd, timeStep_nd)
def_grad_list = len(particleX)*[np.eye(3)]

strain_arr = np.empty([time_nd_arr.shape[0], len(particleX)])
strain_arr[0, :] = 0 # initialize initial strain value

# will contain the x-positions of the particles as it is advected
# easier for plotting 
x_pos_arr = np.empty([time_nd_arr.shape[0], len(particleX)])
z_pos_arr = np.empty([time_nd_arr.shape[0], len(particleZ)])
x_pos_arr[0, :] = particleX
z_pos_arr[0, :] = particleZ

# set-up position during previous time step
with particles.access(particles):
    prev_pos = particles.particle_coordinates.data

cntr = 1
for i, time in enumerate(time_nd_arr[1:]):

    print(time)
    
    # update location of particles
    particles.advection(v_soln.sym, timeStep_nd, order = 2, corrector=False, evalf=False)


    # calculate the strain rate at the particles' current positions
    # with particles.access(grains_str_tensor):
        
    #     # FIXME: uw.function.evaluate() produces a 2x2 tensor, but swarm variable has shape [:, 2, 2] 
    #     # grains_str_tensor.data[:] = uw.function.evaluate(str_rate.sym, particles.data)
    #     # temporary fix - assumes that strain tensor is symmetric
    #     dummy_str_rate = uw.function.evaluate(str_rate.sym, particles.particle_coordinates.data) # evaluate strain rate at current position
    #     grains_str_tensor.data[:, 0] = dummy_str_rate[:, 0, 0] 
    #     grains_str_tensor.data[:, 1] = dummy_str_rate[:, 0, 1]
    #     grains_str_tensor.data[:, 2] = dummy_str_rate[:, 1, 0]
    #     grains_str_tensor.data[:, 3] = dummy_str_rate[:, 1, 1]  

    with particles.access(particles):
        
        pos = particles.particle_coordinates.data # TODO: check if particle index in positions are consistent
        x_pos_arr[i, :] = pos[:, 0]
        z_pos_arr[i, :] = pos[:, 1]
        
        # iterate through particles
        for j, p in enumerate(pos):
            
            p3d = np.append(p, 0.) 

            vg = get_velocity_gradient(np.nan, p3d)

            # create dictionary containing the position at previous and current timesteps of particle
            pos_particle_dict = {time_nd_arr[i - 1]: prev_pos[j],
                                time: pos[j]}
            
            # define get_position function 
            #get_position = lambda t : pos_particle_dict[t]
            get_position = lambda t : prev_pos[j] + ((pos[j] - prev_pos[j])/(time - time_nd_arr[i - 1])) * (t - time_nd_arr[i - 1])

            strain_arr[i, j] = strain_arr[i - 1, j] + _utils.strain_increment(time - time_nd_arr[i - 1], vg)

            def_grad_list[j] = mineral_list[j].update_orientations(
                                                            params,
                                                            def_grad_list[j],
                                                            get_velocity_gradient,
                                                            pathline=(time_nd_arr[i - 1], time, get_position),
                                                        )
    print(f"timestep {i}:", strain_arr[i, :])
    # updates
    prev_pos = pos
    
    time += timeStep_nd
    cntr += 1



# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

vmin = 0
vmax = max(2, strain_arr.max())

fig, ax = plt.subplots(dpi = 300)

for i in range(len(particleX)):
    out = ax.scatter(dim(x_pos_arr[:-1, i], u.meter).magnitude, dim(z_pos_arr[:-1, i], u.meter).magnitude, s = 5, c = strain_arr[:-1, i], vmin = vmin, vmax = vmax, cmap = "viridis")

ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_xlim([0, boxLength])
ax.set_ylim([-boxHeight, 0])
ax.set_aspect("equal")
# ax.set_title(f"Timestep: {cntr}")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
   
plt.colorbar(out, cax = cax)


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


