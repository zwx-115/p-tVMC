
# Neural network quantum states with time evoltuion
This repository contains the code to reproduce parts of the results presented in the paper:

Paths towards time evolution with larger neural-network quantum states

Wenxuan Zhang, Bo Xing, Xiansong Xu, and Dario Poletti

arxiv: https://arxiv.org/pdf/2406.03381

# Installation

The code is written in Julia language, and it depends on the packages below: 

`Flux`, `CUDA`, `LinearAlgebra`, `Statistics`, `Distributions`, `Random`, `Dates`, `JLD2` and `Distributed`

Make sure you have the following installed: Julia 1.7 or later

# Useage

The code includes 2 parts:
### I. Preparation for ground energy:
The folder `ground_energy` includes resource files and scripts, which are used to prepare the ground energy of Ising model(including only the tilted Ising model).

`[Lx]` and `[Ly]` - quamtum system size, for 1D case, `[Ly]` must be set as 1

`BC` - boundary condition of the quantum system.

1. The file `script_gs_fnn.jl` is for preparing the ground state by 3 hidden-layers feed-forward neural network.
2. 
   1.1 `[layer-size]` is the hidden node for each layer and the last hidden layer must be 1
   
   1.2 `[act_func_mode]` is the activatioan functions
   
3. The file `script_gs_rbm.jl` is for preparing the ground state by Restricted Boltzmann machine  neural network.
   
   2.1 `[nh_nv_ratio]` =the hidden density of RBM

### II. Time evolution:
The folder `time_evolution` includes resource files and scripts, which are used for the time evolution of Ising model(including only the tilted Ising model).

1. `script_time_fnn_sr.jl` and `script_time_rbm_ptVMC.jl` -- ptVMC method with SR method for both FNN and RBM

2. `script_time_fnn_tVMC.jl` and `script_time_rbm_tVMC.jl` -- tVMC method for both FNN and RBM

3. `script_time_fnn_minsr.jl` -- ptVMC method with minSR method for FNN

4. `script_time_fnn_KFAC.jl` -- ptVMC method with KFAC method for FNN

5. `script_time_fnn_SOO.jl` -- ptVMC method with KFAC method for FNN

# Features:

### I. sampling method:

1. metropolis method:
   ```
   sample_name = "metropolis" 
   ```
   `[n_thermals]` - number of thermalization
   `[n_sweeps]` - number of sweeps
   `[n_states]` - number of states
   
3. sull summation(all configurations):
   ```
   sample_name = "exact"
   ```
   in this case, the number of states $N_s = 2^{L}$, where $L = L_x *L_y$

### I. options of optimization:

1. Gradient Descent method:
   ```
   update_name = "GD"
   ```
   
2. Adam method:
   ```
   update_name = "Adam"
   ```
   
3. learning rate:

   `[γ]` - initial learning rate
   
   `[final_γ]` - final learning rate
   
   `[decay_rate]` - the decay rate of $\gamma$

   `[decay_interval]` - decay steps, meaning that γ = decay_rate * γ after every decay_interval iterations

   `[n_epochs]` number of epochs

   `[n_loops]` number of loops

4. Trotter decomposition:
   `[evo_setting = ("trotter", n_bonds)]` - trotter method, '[n_bonds]' - the number of bond for each trotter block


   







