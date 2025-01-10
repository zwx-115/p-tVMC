include("./nnqs_gs_fnn.jl")
CUDA.allowscalar(false)
device!(0)
# parameters of quantum model
Lx = 10
Ly = 1
J = 1.0
hx = -0.5
n_sites = Lx *Ly
hz = zeros(n_sites) .- 0.5;
BC = "open"
quant_sys_setting = ("Ising", [Lx,Ly], J, hx, hz, "open")

# parameters of neural network
layer_size = [n_sites, 4n_sites, 3n_sites, 1]

act_func_mode = "lncosh"
seed_num = 1234
#init_type = "gaussian"
init_type = "uniform"
sigma = 5e-2

nn_setting = ("FNN", act_func_mode, layer_size, seed_num, init_type, sigma)


# parameters of sampling

# exact sdampling:
# sample_name = "exact"
# sample_setting = (["exact"])

# metropolis algorithm:
n_thermals = 100
n_sweeps = 1
n_states = 100
sample_setting = ("metropolis", n_thermals, n_sweeps, n_states)




n_epochs = 1
n_loops = 100
block_in = "all"
#block_out = [4n_sites, n_sites, 1]
block_out = "all"
#block_out = [4, 2, 1]
#layer_type = "layer"
layer_type = "all"
overlap_in = 0
overlap_out= [0,0,0,0]
overlap_out = [0,0,0,0]

# parameters of updating
update_name = "GD"
#update_name = "Adam"
γ = 0.05
final_γ = 0.01
decay_rate = 0.8
decay_interval = 10
update_setting = (update_name, n_epochs, n_loops, 
(block_in, block_out), (overlap_in, overlap_out), layer_type,
γ, final_γ, decay_rate, decay_interval)


# parameters of optimization
opt_name = "SR"
#opt_name = "normal"
λ = 1e-2
opt_setting = (opt_name, λ)

# opt_setting = (["normal"])


np = 1

GC.gc()
CUDA.reclaim()

nn_setting = ("FNN", act_func_mode, layer_size, seed_num, init_type, sigma)
neural_network_initialization_run_gpu(ComplexF64,
quant_sys_setting, nn_setting, sample_setting, update_setting, opt_setting, np)
