# This code is for p-tVMC or tVMC

include("./nnqs_time_rbm.jl")
device!(1)
t0 = 0.0
tf = 4.0
dt = 0.1
tol = 1e-5
time_setting = (t0, dt, tf)
#f_gs = load("./Ising, L = 14,  Lx_14,  Ly = 1,  J = 0.0, hx = -1.0.jld2")
f_gs = load("./RBM_GS_14_sites_5_0_163840000_123.jld2") # IM with J = 0, hx = -1.0, hz = 0

NN_parameters = (1, f_gs["RBM_GS_a"], f_gs["RBM_GS_b"], f_gs["RBM_GS_W"])

Lx = 14
Ly = 1
J = 1.0
hx = -0.5
n_sites = Lx *Ly
hz = zeros(n_sites) .- 0.5;
measurment_site = 1

BC = "open"
quant_sys_setting = ("Ising", [Lx,Ly], J, hx, hz, "open", 1)

n_thermals = 100
n_sweeps = 1
n_states = 10000
sample_setting = (n_thermals, n_sweeps, n_states, "Normal")

nn_α = 5
γ = 0.2
seed_num = 1234
neural_net_setting = ("RBM", nn_α, γ, seed_num)


num_epochs = 1
n_loop = 1000
n_shift_site = 0
redu_coff_α = 0.8
final_γ = 0.05
segment_redu_γ = 400
iteration_mode = "SR"

update_setting =("all", [0, 0], num_epochs, n_loop, n_shift_site, redu_coff_α, final_γ, segment_redu_γ, iteration_mode)

n_ham_bond = 5
Neural_Net_Initialization_parallel_Time(ComplexF64, quant_sys_setting, neural_net_setting, sample_setting, update_setting, NN_parameters, time_setting, tol, n_ham_bond)