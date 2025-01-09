# this script is used for 14 spin with p-tVMC and SR method
using Flux
using CUDA
using Statistics
using Distributions 
using Random
using Dates
using JLD
using JLD2
using Distributed
using LinearAlgebra
using Base.Iterators

include("./nnqs_time_model.jl")
include("./nnqs_time_trotter_block.jl")
include("./nnqs_time_local_energy.jl")
include("./nnqs_time_sampling.jl")
include("./nnqs_time_optimization.jl")
include("./nnqs_measurement.jl")
include("./nnqs_main.jl")
device!(1)

# file_time = "/home/users/sutd/1004957/code/nn_time_gpu.jl"
# include(file_time)

t0 = 0.0
tf = 2.0
dt = 0.1
tol = 1e-5
time_setting = (t0, tf, dt)

# the ground energy: paramagnetic phase
# J = 0, hx = -1, hz = 0
fnn_gs = load("./Ground_Energy_FNN_14_sites_16384_exact_14431010_gs.jld2")# IM: 
nn_parameters = (0, 0)

Lx = 14
Ly = 1
J = 1.0
hx = -0.5
n_sites = Lx *Ly
hz = zeros(n_sites) .- 0.5;
measurment_site = 1

BC = "open"
quant_sys_setting = ("Ising", [Lx,Ly], J, hx, hz, "open")

act_func_mode = "lncosh"
seed_num = 1234
init_type = "gaussian"
sigma = 0.05
nn_setting = ("FNN", act_func_mode, fnn_gs["nn_gs_cpu"], seed_num, init_type, sigma, nn_parameters)

sample_name = "metropolis"
if sample_name == "exact"
    sample_setting = (["exact"])
elseif sample_name == "metropolis"
    n_thermals = 100
    n_sweeps = 1
    n_states = 10000
    sample_setting = ("metropolis", n_thermals, n_sweeps, n_states)
end


#update_name = "Adam"
#n_epochs = 1
#n_loops = 2000

# ----------------

block_in = "all"
block_out = "all"
layer_type = "all"
overlap_in = 0
overlap_out= [0,0,0]

# ----------------
# parameters of updating


γ = 0.2
final_γ = 0.05
decay_rate = 0.8
decay_interval = 400
n_epochs = 1
n_loops = 1000
update_setting = ("GD", n_epochs, n_loops, 
(block_in, block_out), (overlap_in, overlap_out), layer_type,
γ, final_γ, decay_rate, decay_interval)

# parameters of optimization

λ = 0.01
α_r = 0.0
opt_setting = ("SR", λ, α_r)

#opt_setting = (["tvmc"])




evo_setting = ("trotter", 5)
#evo_setting = ("rk", 4 ,[1,2,2,1])

CUDA.allowscalar(true)
σ_site = 1
neural_network_initialization_time(quant_sys_setting, nn_setting, 
sample_setting, update_setting, opt_setting, time_setting, tol, 
evo_setting, σ_site)
exit()