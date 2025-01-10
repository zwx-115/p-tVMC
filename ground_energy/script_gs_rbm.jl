include("./nnqs_gs_rbm.jl")
# the GPU card:
device!(0)

Lx = 10
Ly = 1
J = 1.0
hx = -0.5
n_sites = Lx *Ly
hz = zeros(n_sites) .- 0.5;
corr_num = 1
quant_sys_setting =("Ising", [Lx,Ly], J, hx, hz, "open", corr_num)



nh_nv_ratio = 5
γ = 0.1
n_states = 100
n_thermals = 100
n_sweeps = 1
#sample_setting = ("exact", n_thermals, n_sweeps, n_sites)
sample_setting = ("metropolis", n_thermals, n_sweeps, n_states) 


num_epochs = 1

n_loops = 100
n_shift_site = 0
redu_coff_γ = 0.8
final_γ = 0.01

segment_redu_γ = 100

iteration_mode = "SR"
#iteration_mode = "normal_optimization"

exact_value = [-27.029171495081, -54.199820509168, -108.541073823783, -217.223580415040, -135.836910654811]
exact_value_tol = 1e-20
exact_value_setting = (exact_value[1], exact_value_tol)

random_seed_num = 1234


Op_method = "all"


neural_net_setting = ("RBM", nh_nv_ratio, γ, random_seed_num)
update_setting = (Op_method, [0, 0], num_epochs, n_loops, n_shift_site, redu_coff_γ, final_γ, segment_redu_γ, iteration_mode)

println("seed = ", random_seed_num)
Neural_Net_Initialization(ComplexF64, quant_sys_setting, neural_net_setting, sample_setting, update_setting)
