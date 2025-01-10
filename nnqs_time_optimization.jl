# This code is for the optimization:

########### --- optimizing parameters --- ###########
########### --- optimizing parameters --- ###########

abstract type abstract_optimizer end # Define type of activation functions: SR, gradient descent, ...

abstract type abstract_update end # Define type of update: gradient descent, adam, ...

function regularization_SR(S::CuArray{T, 2}, counter::Int; λ₀=100, b=0.9, λ_min = 1e-4) where T
    p = counter
    λ = x -> max(λ₀ * b^p, λ_min)
    Sreg = S + λ(p) * Diagonal(diag(S))
end

function regularization_SR_constant(S::CuArray{T, 2}, λ ) where T
    S += λ * I 
    return S
end


### define type of optimizer
mutable struct sr_optimization{ComplexF64} <: abstract_optimizer
    λ :: Float64
    α_r :: Float64
end

mutable struct min_sr_optimization
    λ :: Float64
end

mutable struct t_vmc{ComplexF64} <: abstract_optimizer
end

### the optimization is normal grident descent with grident force
mutable struct normal_optimization{ComplexF64} 
end

### ------- type of evolution -------
mutable struct rungle_kutta <: abstract_evolution
    order :: Int
    k :: Array{Int,1}
end

#------------------------------------------------------------#

function optimization_params_init(opt_type::sr_optimization, n_params::Int)
    opt_params = Vector(undef, 5)
    opt_params[1] = 0 + 0im  # E_loc_ψϕ_avg
    opt_params[2] = 0 + 0im  # E_loc_ϕψ_avg
    opt_params[3] = CUDA.zeros(ComplexF64, n_params) # ∂θ_avg
    opt_params[4] = CUDA.zeros(ComplexF64, n_params) # E_loc∂θ_avg
    opt_params[5] = CUDA.zeros(ComplexF64, n_params, n_params) # ∂θ_mul_avg
    return opt_params
end

function optimization_params_init(rk::rungle_kutta, n_params::Int)
    opt_params = Vector(undef, 4)
    opt_params[1] = 0 + 0im  # E_loc_avg
    opt_params[2] = CUDA.zeros(ComplexF64, n_params) # ∂θ_avg
    opt_params[3] = CUDA.zeros(ComplexF64, n_params) # E_loc∂θ_avg
    opt_params[4] = CUDA.zeros(ComplexF64, n_params, n_params) # ∂θ_mul_avg
    return opt_params
end

function optimization_params_init(opt_type::normal_optimization, n_params::Int)
    opt_params = Vector(undef, 4)
    opt_params[1] = 0 + 0im  # E_loc_ψϕ_avg
    opt_params[2] = 0 + 0im  # E_loc_ϕψ_avg
    opt_params[3] = CUDA.zeros(ComplexF64, n_params) # ∂θ_avg
    opt_params[4] = CUDA.zeros(ComplexF64, n_params) # E_loc∂θ_avg
    return opt_params
end

# the paper is https://arxiv.org/abs/2302.01941
function optimization_params_init(opt_type::min_sr_optimization, n_params::Int, n_states::Int)
    opt_params = Vector(undef, 3)
    opt_params[1] = CUDA.zeros(ComplexF64, n_states) # E_loc_ψϕ
    opt_params[2] = CUDA.zeros(ComplexF64, n_states) # E_loc_ϕψ
    opt_params[3] = CUDA.zeros(ComplexF64, n_states, n_params) # O
    return opt_params
end


# this is for sampling method
function accum_opt_params!(opt_type::sr_optimization, opt_params_avg::Vector{Any}, 
    E_loc_ψϕ::CuArray{ComplexF64,2}, E_loc_ϕψ::CuArray{ComplexF64,2}, 
    ∂θ::CuArray{ComplexF64,2})

    opt_params_avg[1] += sum(E_loc_ψϕ) # E_loc_ψϕ_avg
    opt_params_avg[2] += sum(E_loc_ϕψ) # E_loc_ϕψ_avg
    opt_params_avg[3] += sum(∂θ, dims = 2) # ∂θ_avg
    opt_params_avg[4] += sum(E_loc_ψϕ .* conj(∂θ), dims = 2) # E_loc∂θ_avg
    opt_params_avg[5] .+= conj(∂θ) * transpose(∂θ) # ∂θ_mul_avg
end


#this is for exact sampling method
function accum_opt_params!(opt_type::sr_optimization, opt_params_avg::Vector{Any}, 
    E_loc_ψϕ::CuArray{ComplexF64,2}, E_loc_ϕψ::CuArray{ComplexF64,2}, 
    ∂θ :: CuArray{ComplexF64,2}, 
    P_ψ::CuArray{Float64,2}, P_ϕ::CuArray{Float64,2})

    opt_params_avg[1] += sum(P_ϕ .* E_loc_ψϕ) # E_loc_ψϕ_avg
    opt_params_avg[2] += sum(P_ψ .* E_loc_ϕψ) # E_loc_ϕψ_avg
    opt_params_avg[3] += sum(P_ϕ .* ∂θ, dims = 2) # ∂θ_avg
    opt_params_avg[4] += sum(P_ϕ .* E_loc_ψϕ .* conj(∂θ) , dims=2)  # E_loc∂θ_avg
    opt_params_avg[5] .+= P_ϕ .*conj(∂θ) * transpose(∂θ) # ∂θ_mul_avg
end

#this is for exact sampling method and for rk method
function accum_opt_params!(rk:: rungle_kutta, opt_type::t_vmc, opt_params_avg::Vector{Any}, 
    E_loc::CuArray{ComplexF64,2}, ∂θ::CuArray{ComplexF64,2}, P_ψ::CuArray{Float64,2})
    opt_params_avg[1] += sum(P_ψ .* E_loc)
    opt_params_avg[2] += sum(P_ψ .* ∂θ, dims = 2)
    opt_params_avg[3] += sum(P_ψ .* E_loc .* conj(∂θ), dims = 2)
    opt_params_avg[4] .+= P_ψ .* conj(∂θ) * transpose(∂θ)
end

# this is for sampling method for normal optimization
function accum_opt_params!(opt_type::normal_optimization, opt_params_avg::Vector{Any}, 
    E_loc_ψϕ::CuArray{ComplexF64,2}, E_loc_ϕψ::CuArray{ComplexF64,2}, 
    ∂θ_avg::CuArray{ComplexF64,1}, E_loc_∂θ_avg::CuArray{ComplexF64,1})
    opt_params_avg[1] += sum(E_loc_ψϕ) # E_loc_ψϕ_avg
    opt_params_avg[2] += sum(E_loc_ϕψ) # E_loc_ϕψ_avg
    opt_params_avg[3] += ∂θ_avg # ∂θ_avg
    opt_params_avg[4] += E_loc_∂θ_avg # E_loc∂θ_avg
end

#this is for markov sampling method and for rk method
function accum_opt_params!(rk:: rungle_kutta, opt_type::t_vmc, opt_params_avg::Vector{Any}, 
    E_loc::CuArray{ComplexF64,2}, ∂θ::CuArray{ComplexF64,2})
    opt_params_avg[1] += sum(E_loc)
    opt_params_avg[2] += sum(∂θ, dims = 2)
    opt_params_avg[3] += sum(E_loc .* conj(∂θ), dims = 2)
    opt_params_avg[4] .+= conj(∂θ) * transpose(∂θ)
end

# this is for markov sampling method and for min sr method
function accum_opt_params!(opt_type::min_sr_optimization, opt_params_avg::Vector{Any}, E_loc_ψϕ::CuArray{ComplexF64,2}, E_loc_ϕψ::CuArray{ComplexF64,2}, ∂θ :: CuArray{ComplexF64,2})
    opt_params_avg[1] .+= transpose(E_loc_ψϕ)
    opt_params_avg[2] .+= transpose(E_loc_ϕψ)
    opt_params_avg[3] .+= transpose(∂θ)
end

function calculate_opt_params(nn_t::FNN_time, opt_type::sr_optimization, opt_params_avg::Vector{Any})
    S = opt_params_avg[5] .- conj(opt_params_avg[3]) * transpose(opt_params_avg[3]) + opt_type.λ * I
    F = - 1 * opt_params_avg[4] * opt_params_avg[2] .- (-1 * opt_params_avg[1] * opt_params_avg[2] * conj(opt_params_avg[3]))
    # θ = [];
    # for k = 1 : nn_t.n_layers - 1
    #     θ = [θ ; nn_t.model[k].b[:]]
    #     θ = [θ ; nn_t.model[k].W[:]]
    # end
    #println("add regularization")
    return S \ F
    #return S \ (F .+ CuArray(opt_type.α_r * θ[:])) 
end

# normal optimization
function calculate_opt_params(opt_type::normal_optimization, opt_params_avg::Vector{Any})
    F = - 1 * opt_params_avg[4] * opt_params_avg[2] .- (-1 * opt_params_avg[1] * opt_params_avg[2] * conj(opt_params_avg[3]))
    return F
end

function calculate_opt_params(opt_type::t_vmc, opt_params_avg::Vector{Any})
    #S = opt_params_avg[4] .- conj(opt_params_avg[2]) * transpose(opt_params_avg[2]) 
    S = opt_params_avg[4] .- conj(opt_params_avg[2]) * transpose(opt_params_avg[2]) + 1e-6 * I
    F = opt_params_avg[3] .- opt_params_avg[1] * conj(opt_params_avg[2])
    #return -im * pinv(S) * F    
    return -im * pinv(S) * F
end

function calculate_opt_params(opt_type::min_sr_optimization, opt_params_avg::Vector{Any})
    n_states = length(opt_params_avg[1])
    println(n_states)
    O_avg = sum(opt_params_avg[3], dims = 1) /n_states
    println("O_avg: ", size(O_avg))
    O = (opt_params_avg[3] .- O_avg) /sqrt(n_states)
    E_ϵ = (opt_params_avg[1] .- sum(opt_params_avg[1])/n_states) /sqrt(n_states) * (sum(opt_params_avg[2])/n_states) 
    #T = O * O' + 0.01 * I
    ∂θ = O' * (-(O * O' + opt_type.λ * I) \ E_ϵ)
    #∂θ = (O' * O + 0.01* I) \ (O' * E_ϵ)
    return ∂θ
    #return IterativeSolvers.cg( S, F )
end


# optimize parameters through markov chain
function optimize_params(::Type{T}, quant_sys::T_model, nn::FNN, nn_t::FNN_time, 
    sample::Sampling, sample_t::sampling_time,
    n_params::Int, Uk::Un, site_m::Array{Int,1}, 
    block_in, block_out, layer_k, opt_type::T3) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64, T3 <: abstract_optimizer}
    opt_params_avg = optimization_params_init(opt_type, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    for j = 1 : sample.n_sweeps
        single_sweep!(quant_sys, nn, sample) # take samples by using Metropolis algorithm
        single_sweep_time!(quant_sys, nn_t, sample_t) # take samples by using Metropolis algorithm
        E_loc_ψϕ = E_local_ψϕ_trotter(quant_sys, nn, nn_t, Uk, sample_t.state_t, site_m)
        E_loc_ϕψ = E_local_ϕψ_trotter(quant_sys, nn, nn_t, Uk, sample.state, site_m)
        calculate_derivatives!(nn_t, sample_t.state_t, sample.n_states, ∂θ, block_in, block_out, layer_k)
        accum_opt_params!(opt_type, opt_params_avg, E_loc_ψϕ, E_loc_ϕψ, ∂θ)
    end
    n_total = sample.n_sweeps * sample.n_states
    opt_params_avg ./= n_total
    ∂W = calculate_opt_params(nn_t, opt_type, opt_params_avg)
    E_avg = abs(-1 * opt_params_avg[1] * opt_params_avg[2] + 1)
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W[:]
end

# train parameters through exact sampling
function optimize_params(::Type{T}, quant_sys::T_model, nn::FNN, nn_t::FNN_time, sample::exact_sampling, sample_t::sampling_time,
    n_params::Int, Uk::Un, site_m::Array{Int,1}, 
    block_in, block_out, layer_k, opt_type::T3) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64, T3 <: abstract_optimizer}
    opt_params_avg = optimization_params_init(opt_type, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    E_loc_ψϕ = E_local_ψϕ_trotter(quant_sys, nn, nn_t, Uk, sample_t.state_t, site_m)
    E_loc_ϕψ = E_local_ϕψ_trotter(quant_sys, nn, nn_t, Uk, sample.state, site_m)
    calculate_derivatives!(nn_t, sample_t.state_t, sample.n_states, ∂θ, block_in, block_out, layer_k)
    #------------------
    #exact_lnψ = activation_func(act_func, nn.b, nn.W, sample.state, nn.n_layers - 1)
    exact_lnψ = nn.model(sample.state)
    # the ratio of first configuration : C_m = exp(ln(ψ(m)) - ln(ψ(x))) where x is the first configuration
    exact_lnψ_ratio = exp.(exact_lnψ .- exact_lnψ[1])
    P_ψ = abs.(exact_lnψ_ratio).^2 / sum(abs.(exact_lnψ_ratio).^2)
    #exact_lnϕ = activation_func(act_func, nn_t.b, nn_t.W, sample_t.state_t, nn_t.n_layers - 1)
    exact_lnϕ = nn_t.model(sample_t.state_t)
    # the ratio of first configuration : C_m = exp(ln(ϕ(m)) - ln(ϕ(x))) where x is the first configuration
    exact_lnϕ_ratio = exp.(exact_lnϕ .- exact_lnϕ[1])
    P_ϕ = abs.(exact_lnϕ_ratio).^2 / sum(abs.(exact_lnϕ_ratio).^2)
    #------------------
    accum_opt_params!(opt_type, opt_params_avg, E_loc_ψϕ, E_loc_ϕψ, ∂θ, P_ψ, P_ϕ)
    ∂W = calculate_opt_params(nn_t, opt_type, opt_params_avg)
    E_avg = abs(-1 * opt_params_avg[1] * opt_params_avg[2] + 1)
    #E_avg = -1 * opt_params_avg[1] * opt_params_avg[2] + 1
    #E_avg_CV = real(E_avg) + 0.5 * (abs( 1 - E_avg^2) - 1)
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W[:]
    #return E_avg_CV, ∂W[:]

end

# optimize parameters through markov chain with normal optimization
function optimize_params(::Type{T}, quant_sys::T_model, nn::FNN, nn_t::FNN_time, 
    sample::Sampling, sample_t::sampling_time,
    n_params::Int, Uk::Un, site_m::Array{Int,1}, 
    block_in, block_out, layer_k, opt_type::normal_optimization) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64}
    opt_params_avg = optimization_params_init(opt_type, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params)
    E_loc_∂θ = CUDA.zeros(ComplexF64, n_params)
    for j = 1 : sample.n_sweeps
        single_sweep!(quant_sys, nn, sample) # take samples by using Metropolis algorithm
        single_sweep_time!(quant_sys, nn_t, sample_t) # take samples by using Metropolis algorithm
        E_loc_ψϕ = E_local_ψϕ_trotter(quant_sys, nn, nn_t, Uk, sample_t.state_t, site_m)
        E_loc_ϕψ = E_local_ϕψ_trotter(quant_sys, nn, nn_t, Uk, sample.state, site_m)
        calculate_derivatives!(nn_t, sample_t.state_t, sample.n_states, E_loc_ψϕ, ∂θ, 
                               E_loc_∂θ, block_in, block_out, layer_k)
        accum_opt_params!(opt_type, opt_params_avg, E_loc_ψϕ, E_loc_ϕψ, ∂θ, E_loc_∂θ)
    end
    n_total = sample.n_sweeps * sample.n_states
    opt_params_avg ./= n_total  
    ∂W = calculate_opt_params(opt_type, opt_params_avg)
    E_avg = abs(-1 * (opt_params_avg[1] * opt_params_avg[2]) + 1)
    overlap = opt_params_avg[1] * opt_params_avg[2] 
    #∂W_new = ∂W .+ (∂W) * real(overlap)
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W[:]
    #return E_avg, ∂W_new[:]

end

# exact_ψ = exp.(activation_func(act_func, nn.b, nn.W, sample.state, nn.n_layers - 1))
# P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)
# exact_ϕ = exp.(activation_func(act_func, nn_t.b, nn_t.W, sample_t.state_t, nn_t.n_layers - 1))
# P_ϕ = abs.(exact_ϕ).^2 / sum(abs.(exact_ϕ).^2)

# train parameters through rk method with markov sampling
function optimize_params(::Type{T}, quant_sys::T_model, nn::FNN,  sample::Sampling,
    n_params::Int, rk::rungle_kutta, opt_type::T3) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64,  T3 <: abstract_optimizer}
    opt_params_avg = optimization_params_init(rk, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    for j = 1 : sample.n_sweeps
        E_loc = calculate_E_local(quant_sys, nn.model, sample.state, sample.n_states, quant_sys.site_bond_y) 
        #∂θ = calculate_derivatives!(nn, sample.state, sample.n_states, ∂θ, "all", "all", "all")
        calculate_derivatives!(nn, sample.state, sample.n_states, ∂θ, "all", "all", "all")
        accum_opt_params!(rk, opt_type, opt_params_avg, E_loc, ∂θ)
    end
    #exact_ψ = exp.(activation_func(act_func, nn.b, nn.W, sample.state, nn.n_layers - 1))
    #exact_ψ = exp.(nn.model(sample.state))
    #P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)
    #accum_opt_params!(rk, opt_type, opt_params_avg, E_loc, ∂θ, P_ψ)
    n_total = sample.n_sweeps * sample.n_states
    opt_params_avg ./= n_total
    ∂W = calculate_opt_params(opt_type, opt_params_avg)
    E_avg = opt_params_avg[1] 
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W[:]
end

# optimize parameters through markov chain and for min_sr method
function optimize_params(::Type{T}, quant_sys::T_model, nn::FNN, nn_t::FNN_time, 
    sample::Sampling, sample_t::sampling_time,
    n_params::Int, Uk::Un, site_m::Array{Int,1}, 
    block_in, block_out, layer_k, opt_type::min_sr_optimization) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64}
    opt_params_avg = optimization_params_init(opt_type, n_params, sample.n_states)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    
    single_sweep!(quant_sys, nn, sample) # take samples by using Metropolis algorithm
    single_sweep_time!(quant_sys, nn_t, sample_t) # take samples by using Metropolis algorithm
    E_loc_ψϕ = E_local_ψϕ_trotter(quant_sys, nn, nn_t, Uk, sample_t.state_t, site_m)
    E_loc_ϕψ = E_local_ϕψ_trotter(quant_sys, nn, nn_t, Uk, sample.state, site_m)
    calculate_derivatives!(nn_t, sample_t.state_t, sample.n_states, ∂θ, block_in, block_out, layer_k)
    accum_opt_params!(opt_type, opt_params_avg, E_loc_ψϕ, E_loc_ϕψ, ∂θ)

    n_total = sample.n_sweeps * sample.n_states
    #opt_params_avg ./= n_total
    ∂W = calculate_opt_params(opt_type, opt_params_avg)
    E_avg = abs(-1 * sum(opt_params_avg[1]) * sum(opt_params_avg[2]) / n_total^2 + 1)
    #println("E_avg = $E_avg")
    opt_params_avg = nothing
    ∂θ = nothing
    #println(∂W[1:2])
    return E_avg, ∂W[:]
end


# optimize parameters : SOO
function optimize_params3(::Type{T}, quant_sys::T_model, nn::FNN, nn_t::FNN_time,sample::Sampling, sample_t::sampling_time, n_params::Int, Uk::Un, site_m::Array{Int,1}, block1_in::T4, block1_out::T4, block2_in::T4, block2_out::T4, block3_in::T4, opt_type::T3) where{T_model<:abstract_quantum_system, T<:ComplexF64, T3 <: abstract_optimizer, T4<:Array{Int}}
    opt_params_avg = optimization_params_init(opt_type, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    for j = 1 : sample.n_sweeps
        single_sweep!(quant_sys, nn, sample) # take samples by using Metropolis algorithm
        single_sweep_time!(quant_sys, nn_t, sample_t) # take samples by using Metropolis algorithm
        E_loc_ψϕ = E_local_ψϕ_trotter(quant_sys, nn, nn_t, Uk, sample_t.state_t, site_m)
        E_loc_ϕψ = E_local_ϕψ_trotter(quant_sys, nn, nn_t, Uk, sample.state, site_m)
        calculate_derivatives3!(nn_t, sample_t.state_t, sample.n_states, ∂θ, block1_in, block1_out, block2_in, block2_out, block3_in)
        accum_opt_params!(opt_type, opt_params_avg, E_loc_ψϕ, E_loc_ϕψ, ∂θ)

    end
    n_total = sample.n_sweeps * sample.n_states
    opt_params_avg ./= n_total
    ∂W = calculate_opt_params(nn_t, opt_type, opt_params_avg)
    #println("E_loc_ψϕ = $(opt_params_avg[1])")
    #println("E_loc_ϕψ = $(opt_params_avg[2])")
    overlap = 1 - opt_params_avg[1] * opt_params_avg[2]
    #println("overlap without square = $(overlap)")
    E_avg = abs(-1 * opt_params_avg[1] * opt_params_avg[2] + 1)
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W[:]
end

#=======================#
#=======================#

########### --- updating parameters --- ###########
########### --- updating parameters --- ###########

mutable struct gradient_descent <: abstract_update
    γ :: Float64
    final_γ :: Float64
    decay_rate::Float64
    decay_interval::Int
end

mutable struct adam <: abstract_update
    γ :: Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m::Vector{Any}
    v::Vector{Any}
end

mutable struct adam_opt <: abstract_update
    γ :: Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m::CuArray
    v::CuArray
    t::Int
end

function adam_opt(γ, β1, β2, ϵ)
    return adam_opt(γ, β1, β2, ϵ, CuArray([0.0 + 0.0im]), CuArray([0.0 + 0.0im]), 1)
end

#----------------------------------------------------

# update all parameters
function update_params!(grad_de::gradient_descent, nn::FNN_time, ∂W::CuArray{ComplexF64,1}, 
    block_in::String, block_out::String, layer_k::String)
    for k = 1 : nn.n_layers - 1
        b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
        b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
        nn.model[k].b .-= grad_de.γ * ∂W[b_ind_s : b_ind_f]
        W_ind_s = b_ind_f + 1
        W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
        nn.model[k].W .-= grad_de.γ * reshape(∂W[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
        # println("layer $k : index of b at start = $b_ind_s")
        # println("layer $k : index of b at final = $b_ind_f")
        # println("layer $k : index of b at start = $W_ind_s")
        # println("layer $k : index of b at final = $W_ind_f")
        # println("==========================================")
    end
end

# update all parameters
function update_params!(grad_de::gradient_descent, nn::FNN_time, ∂W::CuArray{ComplexF64,1}, 
    block_in::String, block_out::String, layer_k::Int)
    
    b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
    b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
    nn.b[k] .-= grad_de.γ * ∂W[b_ind_s : b_ind_f]
    W_ind_s = b_ind_f + 1
    W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
    nn.W[k] .-= grad_de.γ * reshape(∂W[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
    # println("layer $k : index of b at start = $b_ind_s")
    # println("layer $k : index of b at final = $b_ind_f")
    # println("layer $k : index of b at start = $W_ind_s")
    # println("layer $k : index of b at final = $W_ind_f")
    # println("==========================================")

end

# update all parameters with rk method
function update_params!(rk::rungle_kutta, nn::FNN, nn_t0::FNN_time, ∂W::CuArray{ComplexF64,1})where{T1 <: abstract_machine_learning, T2 <: abstract_machine_learning}
    for k = 1 : nn.n_layers - 1
        b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
        b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
        nn.model[k].b .= nn_t0.model[k].b .+ ∂W[b_ind_s : b_ind_f]
        W_ind_s = b_ind_f + 1
        W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
        nn.model[k].W .= nn_t0.model[k].W .+ reshape(∂W[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
        # println("layer $k : index of b at start = $b_ind_s")
        # println("layer $k : index of b at final = $b_ind_f")
        # println("layer $k : index of b at start = $W_ind_s")
        # println("layer $k : index of b at final = $W_ind_f")
        # println("==========================================")
    end
end



# update a part parameters in terms of block_in, block_out, layer_k
function update_params!(grad_de::gradient_descent, nn_t::FNN_time, ∂W::CuArray{ComplexF64,1}, 
        block_in::Array{Int}, block_out::Array{Int}, k::Int)
        nn_t.model[k].b[block_out] .-= grad_de.γ * ∂W[1 : length(block_out)]
        nn_t.model[k].W[block_out, block_in] .-= grad_de.γ * reshape(∂W[length(block_out) + 1 : end], length(block_out), length(block_in))
end

# update all parameters with adam method
function update_params!(adam::adam_opt,  nn::FNN_time, ∂W::CuArray{ComplexF64,1}, 
    block_in::String, block_out::String, layer_k::String)
    #println(adam.m[1:5])
	adam.m = adam.β1 .* adam.m .+ (1 - adam.β1) .* ∂W
    adam.v = adam.β2 .* adam.v .+ (1 - adam.β2) .* (∂W .* conj(∂W))
    m_hat = adam.m ./ (1 - adam.β1^adam.t)
    v_hat = adam.v ./ (1 - adam.β2^adam.t)
    dW = adam.γ .* m_hat ./ (sqrt.(v_hat) .+ adam.ϵ)
    for k = 1 : nn.n_layers - 1
        b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
        b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
        nn.model[k].b[:] .-= dW[b_ind_s : b_ind_f]
        W_ind_s = b_ind_f + 1
        W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
        nn.model[k].W[:,:] .-= reshape(dW[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
    end
end

# SOO:
# update a part parameters in terms of block_in, block_out of all layers
function update_params3!(grad_de::gradient_descent, nn_t::FNN_time, ∂W::CuArray{ComplexF64,1}, block1_in::T, block1_out::T, block2_in::T, block2_out::T, block3_in::T) where{T<:Array{Int}}

    println("size of ∂W is ",size(∂W))
    # first layer:
    b1_ind_f = length(block1_out)
    w1_ind_s = b1_ind_f + 1 
    w1_ind_f = w1_ind_s - 1 + length(block1_out) * length(block1_in)
    nn_t.model[1].b[block1_out] .-= grad_de.γ * ∂W[1 : b1_ind_f]
    nn_t.model[1].W[block1_out,  block1_in] .-= grad_de.γ * reshape(∂W[w1_ind_s : w1_ind_f], length(block1_out), length(block1_in))
    # second layer:
    b2_ind_s = w1_ind_f + 1
    b2_ind_f = b2_ind_s - 1 + length(block2_out)
    w2_ind_s = b2_ind_f + 1 
    w2_ind_f = w2_ind_s - 1 + length(block2_out) * length(block2_in)
    nn_t.model[2].b[block2_out] .-= grad_de.γ * ∂W[b2_ind_s : b2_ind_f]
    nn_t.model[2].W[block2_out, block2_in] .-= grad_de.γ * reshape(∂W[w2_ind_s : w2_ind_f], length(block2_out), length(block2_in))
    # third layer:
    b3_ind_s = w2_ind_f + 1
    b3_ind_f = b3_ind_s 
    w3_ind_s = b3_ind_f + 1 
    w3_ind_f = w3_ind_s - 1 + length(block3_in)
    nn_t.model[3].b .-= grad_de.γ * ∂W[b3_ind_s : b3_ind_f]
    nn_t.model[3].W[[1], block3_in] .-= grad_de.γ * reshape(∂W[w3_ind_s : end], 1, length(block3_in))

end
#=======================#
#=======================#