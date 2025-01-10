# this script is used for 14 spin with p-tVMC and minSR method
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

t0 = 0.0 
tf = 0.2
dt = 0.1 
tol = 1e-2
time_setting = (t0, tf, dt)

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
simu_k = 1 
seed_num = 1234
init_type = "gaussian"
sigma = 0.05
nn_setting = ("FNN", act_func_mode, fnn_gs["nn_gs_cpu"], seed_num, init_type, sigma, nn_parameters)

# ----------------
#parameters of sampling
#sample_name = "exact"
sample_name = "metropolis"
if sample_name == "exact"
    sample_setting = (["exact"])
elseif sample_name == "metropolis"
    n_thermals = 50
    n_sweeps = 1
    n_states = 10000  #!!! number of states to be sampled
    sample_setting = ("metropolis", n_thermals, n_sweeps, n_states)
end



#update_name = "Adam"



# ----------------
# all parameters updatting
block_in = "all"
block_out = "all"
layer_type = "all"
overlap_in = 0
overlap_out= [0,0,0]


# parameters of updating
update_name = "GD"
n_epochs = 1
n_loops = 1000
if update_name == "GD"
    γ = 0.2
    final_γ = 0.05
    decay_rate = 0.8
    decay_interval = 400
    update_setting = (update_name, n_epochs, n_loops, 
    (block_in, block_out), (overlap_in, overlap_out), layer_type,
    γ, final_γ, decay_rate, decay_interval)
elseif update_name == "Adam"
    γ = 1e-4
    β1 = 0.9
    β2 = 0.999
    ϵ = 1e-8
    update_setting = (update_name, n_epochs, n_loops, 
    (block_in, block_out), (overlap_in, overlap_out), layer_type,
    γ, β1, β2, ϵ)
end


# parameters of optimization
#opt_name = "SR"
#opt_name = "tvmc"
opt_name = "minSR"
if opt_name == "SR"
    λ = 0.01
    α_r = 0.0
    opt_setting = ("SR", λ, α_r)
elseif opt_name == "minSR"
    λ = 0.01
    opt_setting = ("minSR", λ)
elseif opt_name == "tvmc"
    opt_setting = (["tvmc"])
end



evo_setting = ("trotter", 5)
#evo_setting = ("rk", 4 ,[1,2,2,1])
σ_site = 1
neural_network_initialization_time(quant_sys_setting, nn_setting, 
sample_setting, update_setting, opt_setting, time_setting, tol, 
evo_setting, σ_site)
exit()