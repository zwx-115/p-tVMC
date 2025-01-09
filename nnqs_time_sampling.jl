# This code if for the Monte Carlo sampling:

###########  --- Sampling --- ###########
###########  --- Sampling --- ###########
abstract type abstract_sampling end # Define sampling methods: Monte Carlo, Metropolis Hsating, ...

mutable struct Sampling <: abstract_sampling
    n_thermals :: Int
    n_sweeps :: Int
    n_states :: Int #This means how many states we updates in one step(The states is a matrix, instead of an vector)
    counter :: Int
    state :: CuArray{Int,2}
end

mutable struct sampling_time <: abstract_sampling
    state_t::CuArray{Int,2}
end


mutable struct exact_sampling <: abstract_sampling
    counter :: Int
    n_states :: Int
    state :: CuArray{Int,2}
end


function thermalization!(quant_sys::T_model, nn ::FNN, nn_t::FNN_time, 
    sample::Sampling, sample_t::sampling_time) where{T_model<:abstract_quantum_system}
    
    sample.state = rand(-1:2:1, quant_sys.n_sites, sample.n_states)
    #sample_t.state_t = rand(-1:2:1, ising.n_sites, sample.n_states)
    for j = 1 : sample.n_thermals
        single_sweep!(quant_sys, nn, sample)
    end
    sample_t.state_t[:,:] .= copy(sample.state[:,:])
end

function gen_flip_list(flip_local::Array{Int,1}, n_sites::Int) # Generate a flip index
    N = length(flip_local)
    shift_num = collect(0: n_sites : (N-1) * n_sites)
    flip_global = flip_local .+ shift_num
    flip_list = [flip_local  flip_global]
    return flip_list
end

# markov chain
function single_sweep!(quant_sys::T_model, nn::FNN, sample::Sampling) where{T_model<:abstract_quantum_system, T2 <: abstract_activation_function}
    #L = quant_sys.n_sites
    #state_new = sample.state
    for j = 1 : quant_sys.n_sites
        x_trial = copy(sample.state)
        #flip_local = Array(rand(1:L, N))
	    flip_local = zeros(Int, sample.n_states) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, quant_sys.n_sites) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        x_trial[j,:] .= -x_trial[j,:]
        p = abs.(ratio_ψ(nn, x_trial, sample.state)).^2
        #println("p= ",p)
        r = CUDA.rand(1,sample.n_states) # generate random numbers 
        #println("r= ",r)
        sr = Array(r.<min.(1,p)) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = findall(x->x==1,sr[:]) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        sample.state[site_ind_accept] .= -sample.state[site_ind_accept]
    end
end

function single_sweep_time!(quant_sys::T_model, nn_t::FNN_time, sample_t::sampling_time) where{T_model<:abstract_quantum_system}
    #L = quant_sys.n_sites
    #state_new = sample_t.state_t
    n_states = size(sample_t.state_t, 2) # Number of chains for each step
    for j = 1 : quant_sys.n_sites
        x_trial = copy(sample_t.state_t)
        #flip_local = CuArray(rand(1:L, N))
	    flip_local = zeros(Int,n_states) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, quant_sys.n_sites) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        x_trial[j,:] .= -x_trial[j,:]
        p = abs.(ratio_ϕ(nn_t, x_trial, sample_t.state_t)).^2
        #println("p= ",p)
        r = CUDA.rand(1,n_states) # generate random numbers 
        #println("r= ",r)
        sr = Array(r.<min.(1,p)) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = findall(x->x==1,sr[:]) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        sample_t.state_t[site_ind_accept] .= -sample_t.state_t[site_ind_accept]
    end
end

#=======================#
#=======================#