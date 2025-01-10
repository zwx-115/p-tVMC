# This code is for 
# 1. calculating loacl energy and local U:
# 2. calculating derivatives of FNN:

########### --- calculate E_loc and derivatives --- ###########
########### --- calculate E_loc and derivatives --- ###########

function E_local_ψϕ_trotter(ising::ising_model, nn::FNN, nn_t::FNN_time, 
    Uk::Un, initial_state::CuArray{Int}, bond_ind::Array{Int,1}) 

    E_ψϕ = CUDA.zeros(ComplexF64, 1, size(initial_state,2))
    local_state = initial_state[bond_ind,:]
    U_col = sum(Int, -0.5*(local_state .- 1) .* Uk.ind_U_col , dims=1) .+ 1 # Convert local state to the number
    H_size = 2^length(bond_ind) 
    for j = 1:H_size
        new_state = copy(initial_state)
        new_state[bond_ind,:] .= Uk.ind_state[:,j]
        #E_ψϕ .+= exp.(activation_func(act_func, nn.b, nn.W, new_state, nn.n_layers - 1) .- 
        #activation_func(act_func, nn_t.b, nn_t.W, initial_state, nn.n_layers - 1)) .* Uk.U[U_col, j]
        E_ψϕ .+= exp.(nn.model(new_state) .- nn_t.model(initial_state)) .* Uk.U[U_col, j]
    end
    return E_ψϕ
end

function E_local_ϕψ_trotter(ising::ising_model, nn::FNN, nn_t::FNN_time, 
    Uk::Un, initial_state::CuArray{Int}, bond_ind::Array{Int,1}) 

    E_ϕψ = CUDA.zeros(ComplexF64,1,size(initial_state,2))
    local_state = initial_state[bond_ind,:]
    U_col = sum(Int, -0.5*(local_state.-1).*Uk.ind_U_col,dims=1) .+ 1 #Convert local state to the number
    H_size = 2^length(bond_ind)
    for j = 1:H_size
        new_state = copy(initial_state)
        new_state[bond_ind,:] .= Uk.ind_state[:,j]
        #E_ϕψ .+=  exp.(activation_func(act_func, nn_t.b, nn_t.W, new_state, nn.n_layers - 1) .- 
        #activation_func(act_func, nn.b, nn.W, initial_state, nn.n_layers - 1)).* Uk.U_dag[U_col, j]
        E_ϕψ .+=  exp.(nn_t.model(new_state) .- nn.model(initial_state)) .* Uk.U_dag[U_col, j]
    end
    return E_ϕψ
end

function calculate_E_local(ising::ising_model, model::Chain, initial_state::CuArray{Int} , n_states::Int, y::Array{Int,1}) # calaulate E_loc for Ising model under 1D case
    E_loc = CUDA.zeros(ComplexF64, 1, n_states)
    E_loc .+= ising.hz' * (initial_state.*1.0) # diagnoal part
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_x[:,1], :] .* initial_state[ising.site_bond_x[:,2],:],dims=1) # diagnoal part for bond x
    #flip_local = zeros(Int, N)
    #flip_list_0 = gen_flip_list(flip_local, ising.n_sites)
    # calculate the off-diagonal part
    #new_state = copy(initial_state)
    for j = 1 : ising.n_sites
        new_state = copy(initial_state)
        new_state[j, :] .= -initial_state[j, :]
        #E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j) .* ising.hx
        #E_loc .+= exp.(calculate_ψ(nn, new_state, act_func) .- calculate_ψ(nn, initial_state, act_func) ).* ising.hx
        #E_loc .+= exp.(calculate_ψ(nn, new_state, act_func) .- calculate_ψ(nn, initial_state, act_func) ).* ising.hx
        E_loc .+= exp.(model(new_state) .- model(initial_state)).* ising.hx

    end
    return E_loc
end

function calculate_E_local(ising::ising_model, model::Chain, initial_state::CuArray{Int} , n_states::Int, y::Array{Int,2}) # calaulate E_loc for Ising model under 2D case
    E_loc = CUDA.zeros(ComplexF64, 1, n_states)
    E_loc .+= ising.hz' * (initial_state.*1.0) # diagnoal part
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_x[:,1], :] .* initial_state[ising.site_bond_x[:,2],:],dims=1) # diagnoal part for bond x
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_y[:,1], :] .* initial_state[ising.site_bond_y[:,2],:],dims=1) # diagnoal part for bond y

    #new_state = copy(initial_state)
    for j = 1 : ising.n_sites
        new_state = copy(initial_state)
        new_state[j, :] .= -initial_state[j, :]
        #E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j) .* ising.hx
        #E_loc .+= exp.(calculate_ψ(nn, new_state, act_func) .- calculate_ψ(nn, initial_state, act_fun) ).* ising.hx
        E_loc .+= exp.(model(new_state) .- model(initial_state)).* ising.hx
    end
    return E_loc
end

function ratio_ψ(nn::FNN, new_states::CuArray{Int}, old_states::CuArray{Int}) 
    #res = exp.(calculate_ψ(fnn, new_state, act_func) .- calculate_ψ(fnn, old_state, act_func))
    res = exp.(nn.model(new_states) .- nn.model(old_states))
    return res
end

function ratio_ψ(model::Chain, new_states::CuArray{Int}, old_states::CuArray{Int}) 
    #res = exp.(calculate_ψ(fnn, new_state, act_func) .- calculate_ψ(fnn, old_state, act_func))
    res = exp.(model(new_states) .- model(old_states))
    return res
end

function ratio_ϕ(nn_t::FNN_time, new_states::CuArray{Int}, old_states::CuArray{Int}) 
    #res = exp.(calculate_ψ(fnn, new_state, act_func) .- calculate_ψ(fnn, old_state, act_func))
    res = exp.(nn_t.model(new_states) .- nn_t.model(old_states))
    return res
end

#=======================#
#=======================#

########### --- Automatic differentiation(AD) --- ###########
########### --- Automatic differentiation(AD) --- ###########

# get full gradients of first layer of neural network
# only computing the first-layer param eters 
# function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray,  
#     block_in::String, block_out::String, layer_k::String)
#     ∂lnψ_∂z = 1.0
#     nn.model[1].σ = nn.diff_func
#     ∂u_∂z_1 = nn.model[1](initial_states)
#     n_bias = length(nn.model[1].b)
#     ∂θ[1 : n_bias, :] .= nn.model[2].W .* ∂u_∂z_1
#     CUDA.@sync begin
#         @cuda blocks = size(initial_states, 2) threads  = 1000 #=
#         =# compute_∂W!(∂θ, nn.model[2].W, ∂u_∂z_1, initial_states)
#     end
#     ∂θ .*= ∂lnψ_∂z
#     nn.model[1].σ = nn.act_func
#     return nothing
# end

# take gradients manually
function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, 
    block_in::String, block_out::String, layer_k::String)
    #println("size of ∂θ = ", size(∂θ))
    for j = 1 : nn.n_layers - 1
        idx_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
        idx_f = idx_s + nn.layer_size[j] * nn.layer_size[j+1] + nn.layer_size[j+1] - 1
        #println("idx_s = ", idx_s, " idx_f = ", idx_f)
        if j == 1
            ∂θ_j = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size, nn.act_func, nn.diff_func)
        elseif j == 2
            ∂θ_j = compute_nn_2nd_gpu(nn, initial_states, collect(1 : nn.layer_size[3]), collect(1 : nn.layer_size[2]))
        elseif j == 3
            ∂lnψ_∂z = 1.0
            n_bias = 1
            ∂θ_j = CUDA.zeros(ComplexF64, nn.layer_size[j] + 1, n_states)
            ∂θ_j[1:n_bias, :] .= 1.0 .+ 0.0im
            ∂θ_j[n_bias + 1 : end, :] .= nn.model[1:2](initial_states)
            ∂θ_j .*= ∂lnψ_∂z
        end
        ∂θ[idx_s : idx_f, : ] .= ∂θ_j
        #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
    end
end

# take a block of gradients manually
function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, 
    block_in :: Array{Int}, block_out::Array{Int}, j ::Int)
    #println("size of ∂θ = ", size(∂θ))
    idx_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
    idx_f = idx_s + nn.layer_size[j] * nn.layer_size[j+1] + nn.layer_size[j+1] - 1
    #println("idx_s = ", idx_s, " idx_f = ", idx_f)
    if j == 1
        ∂θ_j = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size,
                                  block_in, block_out, nn.act_func, nn.diff_func)
    elseif j == 2
        ∂θ_j = compute_nn_2nd_gpu(nn, initial_states, block_out, block_in)
    elseif j == 3
        ∂lnψ_∂z = 1.0
        n_bias = 1
        ∂θ_j = CUDA.zeros(ComplexF64, nn.layer_size[j] + 1, n_states)
        ∂θ_j[1:n_bias, :] .= 1.0 .+ 0.0im
        ∂θ_j[n_bias + 1 : end, :] .= nn.model[1:2](initial_states)
        ∂θ_j .*= ∂lnψ_∂z
    end
    ∂θ .= ∂θ_j
    #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
end

#get full gradients of neural network with pullback and with normal optimization
function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, 
    E_ψϕ::CuArray{ComplexF64, 2},
    ∂θ::CuArray{ComplexF64,1}, E_loc_∂θ::CuArray{ComplexF64,1},
    block_in::String, block_out::String, layer_k::String)
    ps = Flux.params(nn.model)
    #nn_gradients = jacobian(() -> real(nn.model(initial_states)), ps)
    _, back_c = CUDA.@allowscalar Flux.pullback(() -> real(nn.model(initial_states)), ps)
    grad_coeff = CUDA.ones(ComplexF64, 1, n_states)
    for j = 1 : nn.n_layers - 1
        idx_b_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
        idx_b_f = sum(nn.layer_size[2:j+1]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]
        idx_W_s = idx_b_f + 1
        idx_W_f = idx_b_f + nn.layer_size[j] * nn.layer_size[j+1]
        
        ∂θ_j_b = back_c(grad_coeff)[ps[2j]]
        ∂θ_j_W = back_c(grad_coeff)[ps[2j-1]]

        E_loc_∂θ_j_b = back_c(real(E_ψϕ))[ps[2j]] .+ im * back_c(imag(E_ψϕ))[ps[2j]]
        E_loc_∂θ_j_W = back_c(real(E_ψϕ))[ps[2j-1]] .+ im * back_c(imag(E_ψϕ))[ps[2j-1]]
        
        ∂θ[idx_b_s : idx_b_f] .= conj(∂θ_j_b) # 
        ∂θ[idx_W_s : idx_W_f] .= conj(∂θ_j_W[:]) #
        E_loc_∂θ[idx_b_s : idx_b_f] .= E_loc_∂θ_j_b #
        E_loc_∂θ[idx_W_s : idx_W_f] .= E_loc_∂θ_j_W[:]
    end
    return nothing
end

# # get full gradients of neural network
# # computing the all layer parameters 
# function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray,  
#     block_in::String, block_out::String, layer_k::String)
#     # compute the first-layer gradients:
#     ∂lnψ_∂z = 1.0
#     nn.model[1].σ = nn.diff_func
#     ∂u_∂z_1 = nn.model[1](initial_states)
#     n_bias = length(nn.model[1].b)
#     ∂θ[1 : n_bias, :] .= nn.model[2].W[:] .* ∂u_∂z_1
#     CUDA.@sync begin
#         @cuda blocks = size(initial_states, 2) threads  = 1000 #=
#         =# compute_∂W!(∂θ, nn.model[2].W, ∂u_∂z_1, initial_states)
#     end
#     #∂θ .*= ∂lnψ_∂z
#     nn.model[1].σ = nn.act_func
#     n_params_1 = nn.layer_size[1] * nn.layer_size[2] + nn.layer_size[2]
#     # then compute the second layer:
#     ∂lnψ_∂z = 1.0
#     ∂θ[n_params_1 + 1, :] .= 1.0 .+ 0.0im
#     ∂θ[n_params_1 + 2 : end, :] .= nn.model[1](initial_states)
#     ∂θ .*= ∂lnψ_∂z
#     return nothing
# end

# get full gradients of first layer of neural network for rk4 method
function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray,  
    block_in::String, block_out::String, layer_k::String)
    #println("size of ∂θ = ", size(∂θ))
    for j = 1 : nn.n_layers - 1
        idx_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
        idx_f = idx_s + nn.layer_size[j] * nn.layer_size[j+1] + nn.layer_size[j+1] - 1
        #println("idx_s = ", idx_s, " idx_f = ", idx_f)
        if j == 1
            ∂θ_j = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size, nn.act_func, nn.diff_func)
        elseif j == 2
            ∂θ_j = compute_nn_2nd_gpu(nn, initial_states, collect(1 : nn.layer_size[3]), collect(1 : nn.layer_size[2]))
        elseif j == 3
            ∂lnψ_∂z = 1.0
            n_bias = 1
            ∂θ_j = CUDA.zeros(ComplexF64, nn.layer_size[j] + 1, n_states)
            ∂θ_j[1:n_bias, :] .= 1.0 .+ 0.0im
            ∂θ_j[n_bias + 1 : end, :] .= nn.model[1:2](initial_states)
            ∂θ_j .*= ∂lnψ_∂z
        end
        ∂θ[idx_s : idx_f, : ] .= ∂θ_j
        #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
        return nothing
    end
end

# overlap stragety SOO:
# take gradients of the portion of parameters in all layers
function calculate_derivatives3!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, block1_in::Array{Int}, block1_out::Array{Int}, block2_in::Array{Int}, block2_out::Array{Int}, block3_in::Array{Int})
    #println("size of ∂θ = ", size(∂θ))
    #println("idx_s = ", idx_s, " idx_f = ", idx_f)
    
    # take the portion of parameters of the first layer:
    ∂θ_1 = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size, block1_in, block1_out, nn.act_func, nn.diff_func)

    # take the block of parameters of the second layer:
    ∂θ_2 = compute_nn_2nd_gpu(nn, initial_states, block2_out, block2_in)

    # take the block of parameters of third layer:
    ∂lnψ_∂z = 1.0
    n_bias = 1
    ∂θ_3 = CUDA.zeros(ComplexF64, length(block3_in) + 1, n_states)
    ∂θ_3[1:n_bias, :] .= 1.0 .+ 0.0im
    ∂θ_3[n_bias + 1 : end, :] .= nn.model[1:2](initial_states)[block3_in, :]
    ∂θ_3 .*= ∂lnψ_∂z

    ∂θ .= [∂θ_1; ∂θ_2; ∂θ_3]
    #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
end


# get a portion of gradients of neural network
# only computing the first-layer parameters 
# function calculate_derivatives!(nn::FNN_time, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, 
#     block_in :: Array{Int}, block_out::Array{Int}, layer_k::Int) # k means the layer of neural network
#     if layer_k == 1
#         ∂lnψ_∂z = 1.0
#         nn.model[1].σ = nn.diff_func
#         ∂u_∂z_1 = nn.model[1](initial_states)
#         n_bias = length(block_out)
#         n_Weight = length(block_in) * length(block_out)
#         ∂θ[1:n_bias, :] .= nn.model[2].W[block_out] .* ∂u_∂z_1[block_out, :]
#         CUDA.@sync begin
#             @cuda blocks = size(initial_states, 2) threads  = 1000 #=
#             =# compute_∂W!(∂θ, nn.model[2].W[block_out], 
#                             ∂u_∂z_1[block_out, :], 
#                             initial_states[block_in, :])
#         end
#         ∂θ .*= ∂lnψ_∂z
#         nn.model[1].σ = nn.act_func
#     elseif layer_k == 2
#         ∂lnψ_∂z = 1.0
#         n_bias = length(block_out)
#         ∂θ[1:n_bias, :] .= 1.0 .+ 0.0im
#         ∂θ[n_bias + 1 : end, :] .= nn.model[1](initial_states)
#         ∂θ .*= ∂lnψ_∂z
#     end
    
#     return nothing
# end


function compute_nn_1st_gpu(nn::Chain, input::CuArray, ls::Array, f::F1, f_d::F2)where{F1,F2}
    n_states = size(input, 2)
    ∂x₄_∂z₄ = 1.0
    ∂z₄_∂x₃ = repeat(nn[3].W[:], ls[2], n_states)
    nn[2].σ = f_d
    ∂x₃_∂z₃_∂x₂ = repeat(nn[1:2](input), ls[2], 1) .* nn[2].W[:]
    ∂4_∂2 = CUDA.zeros(ComplexF64, ls[2], n_states)
    #println(size(∂4_∂2))
    for m = 1 : ls[2]
        m_ind = (m-1)*ls[3]+1 : m*ls[3]
        ∂4_∂2[m,:] = sum(∂z₄_∂x₃[m_ind,:] .* ∂x₃_∂z₃_∂x₂[m_ind,:], dims=1)
    end
    nn[1].σ = f_d
    ∂x₂_∂z₂ = nn[1](input)
    ∂4_∂1 = CUDA.zeros(ComplexF64, ls[1] * ls[2], n_states)
    for m = 1 : ls[1]
        m_ind = (m-1)*ls[2]+1 : m*ls[2]
        ∂4_∂1[m_ind,:] = ∂4_∂2 .* (transpose(input[m,:]) .* ∂x₂_∂z₂)
    end
    nn[1].σ = f
    nn[2].σ = f
    return [∂4_∂2 .* ∂x₂_∂z₂ ; ∂4_∂1] #[b;W[:]]
end

function compute_nn_1st_gpu(nn::Chain, input::CuArray, ls::Array,
    block_in::Array{Int}, block_out::Array{Int}, f::F1, f_d::F2)where{F1,F2}
    n_out = length(block_out)
    n_in = length(block_in)
    n_states = size(input, 2)
    ∂x₄_∂z₄ = 1.0
    ∂z₄_∂x₃ = repeat(nn[3].W[:], ls[2], n_states)

    nn[2].σ = f_d
    ∂x₃_∂z₃_∂x₂ = repeat(nn[1:2](input), n_out, 1) .* nn[2].W[:,block_out][:]
    ∂4_∂2 = CUDA.zeros(ComplexF64, n_out, n_states)
    #println(size(∂4_∂2))
    for m = 1 : n_out
        m_ind = (m-1)*ls[3]+1 : m*ls[3]
        ∂4_∂2[m,:] = sum(∂z₄_∂x₃[m_ind,:] .* ∂x₃_∂z₃_∂x₂[m_ind,:], dims=1)
    end
    nn[1].σ = f_d
    ∂x₂_∂z₂ = nn[1](input)[block_out,:]
    ∂4_∂1 = CUDA.zeros(ComplexF64, n_in * n_out, n_states)
    for m = 1 : n_in
        m_ind = (m-1)*n_out+1 : m*n_out
        ∂4_∂1[m_ind,:] = ∂4_∂2 .* (transpose(input[block_in[m],:]) .* ∂x₂_∂z₂)
    end
    nn[1].σ = f
    nn[2].σ = f
    return [∂4_∂2 .* ∂x₂_∂z₂ ; ∂4_∂1] #[b;W[:]]
end

function compute_nn_2nd_gpu(nn::FNN_time, initial_states::CuArray,block_out::Array{Int}, block_in::Array{Int})
    ∂lnψ_∂z = 1.0 
    nn.model[2].σ = nn.diff_func
    ∂u_∂z_1 = nn.model[1:2](initial_states)
    n_bias = length(block_out) 
    n_Weight = length(block_in) * length(block_out)
    ∂θ = CUDA.zeros(ComplexF64, n_bias + n_Weight, size(initial_states, 2))
    ∂θ[1:n_bias, :] .= nn.model[3].W[block_out] .* ∂u_∂z_1[block_out, :]
    CUDA.@sync begin
        @cuda blocks = size(initial_states, 2) threads  = 1000 #=
        =# compute_∂W!(∂θ, nn.model[3].W[block_out], 
                        ∂u_∂z_1[block_out, :], 
                        nn.model[1](initial_states)[block_in, :])
    end
    #println("======")
    ∂θ .*= ∂lnψ_∂z
    nn.model[2].σ = nn.act_func
    return ∂θ
end

function compute_∂W!(∂θ, W2, ∂u_∂z, x) # for loop is number of column: size(W,2)
    ind_th = threadIdx().x   # thread means different rows: i
    ind_b = blockIdx().x     # Block means different sites: k-th site
    dim_th = blockDim().x    #number of threads
    dim_b = gridDim().x      #number of blocks
    n2 = length(W2)          # number of bias and the number of rows of W1: size(W1,1)
    for m = ind_th : dim_th : length(W2) # the i script of W_ij 
        for n = 1 : size(x, 1) # the j script of W_ij
            ∂θ[n * n2 + m , ind_b] = W2[m] * ∂u_∂z[m, ind_b] * x[n, ind_b]
            # ∂θ[(n-1) * n2 + m + n2, ind_b]
        end
    end
    return nothing
end

#=======================#
#=======================#
