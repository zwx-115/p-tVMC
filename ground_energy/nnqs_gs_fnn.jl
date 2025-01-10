#### This is feedforward neuralnetwork for finding ground state by gpu
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


abstract type abstract_quantum_system end # Define type of quantum systems: Ising model, Heisenberg models, ...

abstract type abstract_machine_learning{T <: Number} end # Define neural network models: Restricted Boltzmann Machine, Convetional Neural Network ...

abstract type abstract_sampling end # Define sampling methods: Monte Carlo, Metropolis Hsating, ...

abstract type abstract_activation_function end # Define type of activation functions: Relu, Leakey Relu, ...

abstract type abstract_optimizer end # Define type of activation functions: SR, gradient descent, ...

abstract type abstract_update end # Define type of update: gradient descent, adam, ...

########### ---Neural Network Models --- ###########
########### ---Neural Network Models --- ###########

# define feed forward neural network
mutable struct FNN{T} <: abstract_machine_learning{T}
    model
    n_layers::Int
    layer_size::Array{Int,1}
    act_func
    f_diff
end

mutable struct DenseC #neural network with complex parameters
    W
    b
    σ
end

mutable struct DenseC_c #neural network with complex parameters
    W
    b
    σ
end

function DenseC((in, out)::Pair{<:Integer, <:Integer}, σ, d; bias = true) 
    W = CuArray(rand(d, out, in).+ im * rand(d, out, in))
    #W = CuArray(rand( out, in).+ im * rand( out, in)) .* 0.001
    #b = Flux.create_bias(W, bias, out)
    b = bias ? CuArray(rand(d, out) .+ im * rand(d, out)) : CUDA.zeros(out)
    DenseC(W, b, σ)
end


function DenseC_c((in, out)::Pair{<:Integer, <:Integer}, σ, d; bias = true) 
    W = rand(d, out, in).+ im * rand(d, out, in)
    b = Flux.create_bias(W, bias, out)
    b = bias ? rand(d, out) .+ im * rand(d, out) : zeros(out)
    DenseC_c(W, b, σ)
end

function (a::DenseC)(x)
    #σ = a.σ
    a.σ(a.W * x .+ a.b)
end

function (a::DenseC_c)(x)
    #σ = a.σ
    a.σ(a.W * x .+ a.b)
end

Flux.@functor DenseC
Flux.@functor DenseC_c

# -------------------------------------------------------------

function FNN_initialization(::Type{ComplexF64}, n_layers::Int, layer_size::Array{Int,1}, act_func, f_diff, random_seed_num::Int; 
    distribution_type::String = "gaussian", sigma::Float64 = 0.05)  
    println("sigma = ", sigma)
    if distribution_type == "gaussian"
        Random.seed!(random_seed_num)
        d = Normal(0, sigma)
        println("gaussian distribution!")
    elseif distribution_type == "uniform"
        Random.seed!(random_seed_num)
        d = Uniform(-sigma, sigma)
        println("uniform distribution!")
    end
    layer = [];
    for k = 1 : n_layers - 1
        if k != n_layers - 1
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], act_func, d)]
        elseif k == n_layers - 1
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], z -> z, d)]
            #layer = [layer; DenseC(layer_size[k] => layer_size[k+1], act_func, d)]
        end
    end
    model = Chain(Tuple(layer))
    return FNN{ComplexF64}(model, n_layers, layer_size, act_func, f_diff)
end

function FNN_initialization_cpu(::Type{ComplexF64}, n_layers::Int, layer_size::Array{Int,1}, act_func, f_diff,random_seed_num::Int; distribution_type::String = "gaussian", sigma::Float64 = 0.05)  where{F}
    if distribution_type == "gaussian"
        Random.seed!(random_seed_num)
        d = Normal(0, sigma)
    end
    layer = [];
    for k = 1 : n_layers - 1
        if k != n_layers - 1
            layer = [layer; DenseC_c(layer_size[k] => layer_size[k+1], act_func, d)]
        elseif k == n_layers - 1
            layer = [layer; DenseC_c(layer_size[k] => layer_size[k+1], z -> z, d)]
        end
    end
    model = Chain(Tuple(layer))
    return FNN{ComplexF64}(model, n_layers, layer_size, act_func, f_diff)
end

#=======================#
#=======================#

########### --- Quantum Systems --- ###########
########### --- Quantum Systems --- ###########

mutable struct ising_model <: abstract_quantum_system
    J :: Float64
    hx :: Float64
    hz :: CuArray{Float64,1}
    n_sites :: Int
    site_bond_x :: Array{Int,2}
    site_bond_y :: Array{Int}
end
function ising_model_initialization(J::Float64, hx::Float64, hz::Array{Float64, 1}, n_sites_x::Int, n_sites_y::Int, BC::String) #BC means Boundary conditions
    n_sites = n_sites_x * n_sites_y

    if BC == "open" #open boundary condition
        x = 0
    elseif BC == "closed" #closed boundary condition
        x = 1
    else 
        error("No Such boundary Condition: $BC.  The option of boundary condition are: closed and open")
    end
    site_bond_x, site_bond_y = configuration_initialization(n_sites_x, n_sites_y, Val(x))
    return ising_model(J, hx, CuArray(hz), n_sites, site_bond_x, site_bond_y)

end

function configuration_initialization(x::Int , y::Int , :: Val{1} ) # Val{1} means the closed boundary condition
    if (x == 0 || y == 0) error("Wrong congfiguration setting: Number of site on X direction: $x,  Number of site on Y direction: $y") end
    Site_Bond_x = zeros(Int, x*y, 2)
    Site_Bond_y = zeros(Int, x*y, 2)
    c = 0
    for yn = 1 : y 
        for xn = 1 : x
            c = c+1
            m = (yn-1) * x + xn

            Site_Bond_x[c,1] = m
            Site_Bond_x[c,2] = mod(m,x) + 1  + (yn-1) * x  #bond connection along x direction 

            Site_Bond_y[c,1] = m
            if yn != y
                Site_Bond_y[c,2] = m + x  #bond connection along y direction 
            else
                Site_Bond_y[c,2] = xn
            end
        end
    end
    if y == 1 
        Site_Bond_y = [0] # Under 1D case, no y direction. 
    elseif y == 2
        Site_Bond_y = Site_Bond_y[1:x,:] #For a ladder configuration, no periodic boundary condition along y direction
    end
    return Site_Bond_x, Site_Bond_y
end

# Val{0} means the open boundary condition: the last site cannot connect to the first site, and here we just consider the nearest-neighbor interaction
function configuration_initialization(x::Int , y::Int , :: Val{0} ) # Val{0} means the open boundary condition

    if (x == 0 || y == 0) error("Wrong congfiguration setting: Number of site on X direction: $x,  Number of site on Y direction: $y") end
    Site_Bond_x = zeros(Int, (x-1)*y , 2)
    cx = 0
    for yn = 1 : y 
        for xn = 1 : x - 1
            m = (yn-1) * x + xn
            cx = cx + 1
            Site_Bond_x[cx,1] = m
            Site_Bond_x[cx,2] = m + 1 #bond connection along x direction
        end
    end
    
    if y >= 2
        cy = 0
        Site_Bond_y = zeros(Int, (y-1)*x , 2)
        for yn = 1 : y - 1 
            for xn = 1 : x 
                m = (yn-1) * x + xn
                cy = cy + 1
                Site_Bond_y[cy,1] = m
                Site_Bond_y[cy,2] = m + x #bond connection along y direction
            end
        end
    else 
        Site_Bond_y = [0]
    end

    return Site_Bond_x, Site_Bond_y
end

function generate_block_2D(Lx :: T, Ly :: T, block_size:: Array{T,1}) where{T<:Int}
	block_size_x = block_size[1]
	block_size_y = block_size[2]
	N_blocks_x = ceil(Int,Lx/block_size_x)
	N_blocks_y = ceil(Int,Ly/block_size_y)
	Block_ind = []
	for j = 1 : 2 * N_blocks_y -1
		block_size_overlap_y = Int(block_size_y/2)
		block_ind_y = collect( (j-1)*block_size_overlap_y + 1 : block_size_overlap_y * (j+1) )
		block_ind_y = (block_ind_y .-1).* Lx .+1
		for k = 1 : 2 * N_blocks_x - 1
			Block_xy = Array{Int}(undef,0)
			block_size_overlap_x = Int(block_size_x/2)
			for k_y = 1 : length(block_ind_y)
				Block_local_xy = block_ind_y[k_y] .+ collect( (k-1)*block_size_overlap_x + 1 : block_size_overlap_x * (k+1)  ) .-1
				Block_xy = [Block_xy; Block_local_xy]
			end
			push!(Block_ind, Block_xy)
		end
	end
	return Block_ind
end

function generate_block_2D(quan_sys :: abstract_quantum_system) # The case where the block size in x = 2, and in y = 1 
	Block_ind = []
	for j = 1 : length(quan_sys.site_bond_x[:,1])
		push!(Block_ind, quan_sys.site_bond_x[j,:] )
	end

	for j = 1 : length(quan_sys.site_bond_y[:,1])
		push!(Block_ind, quan_sys.site_bond_y[j,:] )
	end
	return Block_ind

end	

# generate a list of blocks
function generate_blocks(layer_k_size::Int, block_size::Int, overlap::Int)
    block_list = [];
    n_shift = block_size - overlap
    break_val = 0
    for i = 1 : n_shift : layer_k_size 
        if i + block_size - 1 <= layer_k_size
            block_list = push!(block_list, collect(i : i + block_size - 1))
        else
            block_list = push!(block_list, collect(i : layer_k_size))
        end
        if i + block_size - 1 >= layer_k_size
            break
        end
    end
    block_list_reverse = reverse(block_list[2:end-1])
    #block_list_reverse = reverse(block_list[1:end])

    
    if length(block_list) == 1 
        return block_list
    else
        return [block_list; block_list_reverse]
    end
end

#=======================#
#=======================#

########### --- activation function --- ###########
########### --- activation function --- ###########

function activation_function(activation::String)
    if activation == "relu2C"
        return f(z) = z ./ abs.(z) .* max.(real.(z) .+ imag.(z), 0)
    elseif activation == "lncosh"
        return g(z) = z .- z.^3/3 .+ z.^5 * 2/15 
    elseif activation == "lncosh_exact"
        return p(z) = log.(cosh.(z))
    elseif activation == "lncosh2"
        return m(z) = 0.5 * z.^2 - 1/12 * z.^4 #+ 1/720 * z.^6
    elseif activation == "relu3C2"
        return h(z) = z ./ abs.(z)
    elseif activation == "tanhC"
        return k(z) = tanh.(real.(z)) .+ im .* tanh.(imag.(z))
    elseif activation == "lnrbm"
        return l(z) = reduce(*, exp.(z) .+ exp.(-z))
    else
        error("Wrong activation function")
    end
end

function activation_function_diff(activation::String)
    if activation == "lncosh"
        return g(z) = 1 .- z.^2 .+ z.^4 * 2/3 
    elseif activation == "lncosh_exact"
        return p(z) = tanh.(z)
    end
end

#=======================#
#=======================#

########### --- Local Energy: E_loc --- ###########
########### --- Local Energy: E_loc --- ###########


function calculate_E_local(ising::ising_model, model::Chain, initial_state::CuArray{Int} , n_states::Int, y::Array{Int,1}) # calaulate E_loc for Ising model under 1D case
    E_loc = CUDA.zeros(ComplexF64, 1, n_states)
    E_loc .+= ising.hz' * (initial_state.*1.0) # diagnoal part
    #print("diagonal element of hz: ",E_loc)
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_x[:,1], :] .* initial_state[ising.site_bond_x[:,2],:],dims=1) # diagnoal part for bond x
    #print("diagonal element of J: ",E_loc)

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
    #print("off diagonal element: ",E_loc)

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

function ratio_ψ(model::Chain,  new_states::CuArray{Int}, old_states::CuArray{Int}) 
    #res = exp.(calculate_ψ(fnn, new_state, act_func) .- calculate_ψ(fnn, old_state, act_func))
    res = exp.(model(new_states) .- model(old_states))
    return res
end


#=======================#
#=======================#

########### --- Automatic differentiation(AD) --- ###########
########### --- Automatic differentiation(AD) --- ###########

# get full gradients of neural network
# function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, np::Int, 
#     block_in::String, block_out::String, layer_k::String)
#     ps = Flux.params(nn.model)
#     n_block_x = Int(n_states / np) # the number of states for each thread / block
#     save_back = Vector{Any}(undef, np)  # save back function for each thread: np means number of threads
#     for n = 1 : np
#         _, back_n = Flux.pullback(() ->nn.model(initial_states[:, (n-1) * n_block_x + 1 : n * n_block_x]), ps)
#         save_back[n] = back_n
#     end
#     #v_matrix = cu(1 * Matrix(I, n_block_x, n_block_x))
#     @sync begin 
#         Threads.@threads for pid = 1 : np # pid means the id of thread and each loop compute a blocks of x
#             for i = 1 : n_block_x
#                 v = zeros(1, n_block_x)
#                 v[i] = 1
#                 v = cu(v)
#                 for j = 1 : nn.n_layers - 1
#                     idx_b_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
#                     idx_b_f = sum(nn.layer_size[2:j+1]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]
#                     idx_W_s = idx_b_f + 1
#                     idx_W_f = idx_b_f + nn.layer_size[j] * nn.layer_size[j+1]

#                     du_b = save_back[pid](CUDA.ones(1,n_states))[ps[2j]][:] # bias b
#                     dv_b = save_back[pid](v*im)[ps[2j]][:] 
#                     du_W = save_back[pid](CUDA.ones(1,n_states))[ps[2j-1]][:] # weight W
#                     dv_W = save_back[pid](v*im)[ps[2j-1]][:]
#                     ∂b = (conj(du_b) .+ im * conj(dv_b)) ./ 2
#                     ∂W = (conj(du_W) .+ im * conj(dv_W)) ./ 2
#                     #∂b = (conj(du_b) .+ im * conj(du_b*im)) ./ 2
#                     #∂W = (conj(du_W) .+ im * conj(du_W*im)) ./ 2
#                     ∂θ[idx_b_s : idx_b_f, (pid - 1) * n_block_x + i] .= ∂b # bias b
#                     ∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= ∂W # weight W
#                     #∂θ[idx_b_s : idx_b_f, (pid - 1) * n_block_x + i] .= du_b # bias b
#                     #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
#                 end
#             end
#         end
#     end
# end


# #get full gradients of neural network
# function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, np::Int, 
#     block_in::String, block_out::String, layer_k::String)
#     n_block_x = Int(n_states / np) # the number of states for each thread / block
#     ps = Flux.params(nn.model)
#     #nn_gradients = jacobian(() -> real(nn.model(initial_states)), ps)
#     #@sync begin
#     #    Threads.@threads for pid = 1 : np
#         pid = 1
#             x_i = (pid - 1) * n_block_x + 1 # index of first configureation in the pid-th block
#             x_f = pid * n_block_x # # index of last configureation in the pid-th blockk
#             nn_gradients = CUDA.@allowscalar Flux.jacobian(() -> real(nn.model(initial_states[:, x_i : x_f])), ps)
#             for j = 1 : nn.n_layers - 1
#                 idx_b_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
#                 idx_b_f = sum(nn.layer_size[2:j+1]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]
#                 idx_W_s = idx_b_f + 1
#                 idx_W_f = idx_b_f + nn.layer_size[j] * nn.layer_size[j+1]

#                 ∂θ_j_b = nn_gradients[Flux.params(nn.model[j])[2]]
#                 ∂θ_j_W = nn_gradients[Flux.params(nn.model[j])[1]] 

                
#                 ∂θ[idx_b_s : idx_b_f, x_i : x_f] .= ∂θ_j_b' # transpose(cnoj(b))
#                 ∂θ[idx_W_s : idx_W_f, x_i : x_f] .= ∂θ_j_W' # transpose(cnoj(W))
#                 #∂θ[idx_b_s : idx_b_f, (pid - 1) * n_block_x + i] .= du_b # bias b
#                 #∂θ[idx_W_s : idx_W_f, (pid - 1) * n_block_x + i] .= du_W # weight W
#             end
#     #    end
#    #end

# end

# -------------------------------------------

# take gradients of any hidden layers manually
function calculate_derivatives_any_hidden!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray)
    # ∂f_∂x of last hidden layer
    ind_last_param = length(nn.layer_size ) -1
    ∂f_∂x_k = repeat(nn.model[ind_last_param].W, 1, 1, n_states)
    # k means the k-th parmeters between two hidden layers
    for k = length(nn.layer_size) - 2 : - 1 : 1
        #println("k = $k")
        n_params_k = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]
        #idx_f = idx_s + nn.layer_size[k] * nn.layer_size[k+1] + nn.layer_size[k+1] - 1
        # compute ∂x/∂z
        nn.model[k].σ = nn.f_diff
        ∂x_∂z =  reshape(nn.model[1:k](initial_states), 1, nn.layer_size[k+1], n_states)
        nn.model[k].σ = nn.act_func

        # compute ∂f/∂x * ∂x/∂z
        # ∂f_∂z = ∂b
        ∂f_∂z = reshape(∂f_∂x_k .* ∂x_∂z, nn.layer_size[k+1],  n_states)
        #println("size of ∂f_∂z = ", size(∂f_∂z))
        # compute ∂z/∂W and ∂f/∂W
        if k >= 2
            input = nn.model[1 : k-1](initial_states)
        else
            input = copy(initial_states)
        end
        #println("size of input = ", size(input))
        ∂θ[n_params_k + 1 : n_params_k + nn.layer_size[k+1] , : ] .= ∂f_∂z
        for m = 1 : nn.layer_size[k]       
            #∂f_∂W, ∂f_∂z .* transpose(input[m, :])
            idx_s_m = n_params_k + nn.layer_size[k+1] + nn.layer_size[k+1] * (m-1) + 1 
            idx_f_m = n_params_k + nn.layer_size[k+1] + nn.layer_size[k+1] * (m)
            ∂θ[idx_s_m : idx_f_m, : ] .= ∂f_∂z .* transpose(input[m, :])
        end
        #∂θ[idx_s : idx_f, : ] .= [∂f_∂z; ∂f_∂W]

        # update ∂f/∂x_k to (k-1)-th hiddenlayer
        
        ∂x_k =  ∂x_∂x_pre(nn, initial_states, n_states, k)
        ∂f_∂x_k = matmul_dims_3(∂f_∂x_k, ∂x_k) 

        #∂f_∂x_k = ∂f_∂x_k * ∂x_k

    end
    # compute ∂f/∂W of last hidden layer
    k = length(nn.layer_size) - 1
    idx_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k] + 1
    idx_f = idx_s + nn.layer_size[k] * nn.layer_size[k+1] + nn.layer_size[k+1] - 1
    h_ind = nn.layer_size[end-1]
    ∂θ[idx_s, : ] .= 1.0 + 0.0im # ∂b of last hidden layer
    ∂θ[idx_s + 1 : end, : ] .= nn.model[1 : k - 1](initial_states)
    return nothing
end

function ∂x_∂x_pre(nn::FNN, initial_states::CuArray, n_states::Int, k::Int)
    nn.model[k].σ = nn.f_diff
    ∂x_∂z = reshape(nn.model[1:k](initial_states), nn.layer_size[k+1], 1, n_states) # ∂x_{k+1}/∂z_{k+1}
    # ∂z_∂x_pre = nn[k].W
    nn.model[k].σ = nn.act_func
    return ∂x_∂x_pre = ∂x_∂z .* nn.model[k].W
end

function matmul_dims_3(a::CuArray{ComplexF64, 3}, b::CuArray{ComplexF64,3})
    # compute a * b at the first two dimension
    n_row = size(a, 1)
    n_col = size(b, 2)
    n_states = size(a,3)
    c = CuArray(zeros(ComplexF64, n_row, n_col, n_states))
    for i = 1 : n_row
        for j = 1 : n_col
            c[i,j,:] .= transpose(sum(a[i, :, :] .* b[:, j , :], dims = 1))
        end
    end
    return c
end
# -------------------------------------------
# take gradients manually
function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, np::Int, 
    block_in::String, block_out::String, layer_k::String)
    #println("size of ∂θ = ", size(∂θ))
    for j = 1 : nn.n_layers - 1
        idx_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
        idx_f = idx_s + nn.layer_size[j] * nn.layer_size[j+1] + nn.layer_size[j+1] - 1
        #println("idx_s = ", idx_s, " idx_f = ", idx_f)
        if j == 1
            ∂θ_j = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size, nn.act_func, nn.f_diff)
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
function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::CuArray, np::Int, 
    block_in :: Array{Int}, block_out::Array{Int}, j ::Int)
    #println("size of ∂θ = ", size(∂θ))
    idx_s = sum(nn.layer_size[2:j]) + nn.layer_size[1:j-1]' * nn.layer_size[2:j]+ 1
    idx_f = idx_s + nn.layer_size[j] * nn.layer_size[j+1] + nn.layer_size[j+1] - 1
    #println("idx_s = ", idx_s, " idx_f = ", idx_f)
    if j == 1
        ∂θ_j = compute_nn_1st_gpu(nn.model, initial_states, nn.layer_size,
                                  block_in, block_out, nn.act_func, nn.f_diff)
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




# get one-layer or a blocks of gradients of neural network
# function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::AbstractArray, np::Int, 
#     block_in :: Array{Int}, block_out::Array{Int}, k ::Int) # k means the layer of neural network
#     ps = Flux.params(nn.model[k])
#     n_block_x = Int(n_states / np) # the number of states for each thread / block
#     save_back = Vector{Any}(undef, np)  # save back function for each thread: np means number of threads
#     #for n = 1 : np
#         _, back_n = Flux.pullback(() ->nn.model(initial_states[:, (n-1) * n_block_x + 1 : n * n_block_x]), ps)
#     #    save_back[n] = back_n
#     #end
#     #@sync begin 
#         #Threads.@threads for pid = 1 : np # pid means the id of thread and each loop compute a blocks of x
#         for pid = 1 : np
#             for i = 1 : n_block_x 
#                 v = zeros(1, n_block_x)
#                 v[i] = 1
#                 v = cu(v)
#                 du_b = save_back[pid](v)[ps[2]][block_out]
#                 #dv_b = save_back[pid](v * im)[ps[2]][block_out]
#                 du_W = save_back[pid](v)[ps[1]][block_out , block_in]
#                 #dv_W = save_back[pid](v * im)[ps[1]][block_out , block_in]
#                 #∂b = (conj(du_b) .+ im * conj(dv_b)) ./ 2
#                 #∂W = (conj(du_W) .+ im * conj(dv_W)) ./ 2
#                 ∂b = (conj(du_b) .+ im * conj(du_b*im)) ./ 2
#                 ∂W = (conj(du_W) .+ im * conj(du_W*im)) ./ 2
#                 ∂θ[:, (pid - 1) * n_block_x + i] .= [∂b ; ∂W[:]]
#                 #∂θ[:, (pid - 1) * n_block_x + i] .= [du_b ; du_W[:]]
#             end
#         end
#     #end
# end

# taking gradients by jacobian function
# function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::AbstractArray, np::Int, 
#     block_in :: Array{Int}, block_out::Array{Int}, k ::Int) # k means the layer of neural network
#     ps = Flux.params(nn.model[k])
#     n_block_x = Int(n_states / np) # the number of states for each thread / block
#     W_col = size(nn.model[k].W , 1)
#     W_index = repeat((block_in' .- 1) * W_col, length(block_out)) .+ block_out
#     W_index = W_index[:]
#     #@sync begin
#         #Threads.@threads for pid = 1 : np
#             #Threads.@spawn begin
#                 pid = 1
#                 x_i = (pid - 1) * n_block_x + 1 # index of first configureation in the pid-th block
#                 x_f = pid * n_block_x # # index of last configureation in the pid-th blockk
#                 nn_gradients = CUDA.@allowscalar Flux.jacobian(() -> real(nn.model(initial_states[:, x_i : x_f])), ps)
#                 ∂θ_j_b = nn_gradients[Flux.params(nn.model[k])[2]] # bias b
#                 ∂θ_j_W = nn_gradients[Flux.params(nn.model[k])[1]] # weight W  
#                 ∂θ[:, x_i : x_f] .= [∂θ_j_b[:, block_out]' ; ∂θ_j_W[:, W_index]']# transpose(cnoj(b))
#             #end
#         #end
#     #end
#      # transpose(cnoj(W))
#      return nothing
# end

# taking gradients analytically
# function calculate_derivatives!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, ∂θ::AbstractArray, np::Int, 
#     block_in :: Array{Int}, block_out::Array{Int}, k ::Int) # k means the layer of neural network
#     ps = Flux.params(nn.model[k])
#     n_block_x = Int(n_states / np) # the number of states for each thread / block
#     W_col = size(nn.model[k].W , 1)
#     W_index = repeat((block_in' .- 1) * W_col, length(block_out)) .+ block_out
#     W_index = W_index[:]
#     #@sync begin
#         #Threads.@threads for pid = 1 : np
#             #Threads.@spawn begin
#                 pid = 1
#                 x_i = (pid - 1) * n_block_x + 1 # index of first configureation in the pid-th block
#                 x_f = pid * n_block_x # # index of last configureation in the pid-th blockk
#                 nn_gradients = CUDA.@allowscalar Flux.jacobian(() -> real(nn.model(initial_states[:, x_i : x_f])), ps)
#                 ∂θ_j_b = nn_gradients[Flux.params(nn.model[k])[2]] # bias b
#                 ∂θ_j_W = nn_gradients[Flux.params(nn.model[k])[1]] # weight W  
#                 ∂θ[:, x_i : x_f] .= [∂θ_j_b[:, block_out]' ; ∂θ_j_W[:, W_index]']# transpose(cnoj(b))
#             #end
#         #end
#     #end
#      # transpose(cnoj(W))
#      return nothing
# end



function calculate_derivatives2!(nn::FNN, initial_states::CuArray{Int}, n_states::Int, 
    ∂θ::AbstractArray, E_loc∂θ::AbstractArray, E_loc::CuArray{ComplexF64,2}, np::Int, 
    block_in :: Array{Int}, block_out::Array{Int}, k ::Int) # k means the layer of neural network
    ps = Flux.params(nn.model[k])
    n_block_x = Int(n_states / np) # the number of states for each thread / block
    W_col = size(nn.model[k].W , 1)
    W_index = repeat((block_in' .- 1) * W_col, length(block_out)) .+ block_out
    W_index = W_index[:]
    #@sync begin
        #Threads.@threads for pid = 1 : np
            #Threads.@spawn begin
                #pid = 1
            _ , back_j = Flux.pullback(() -> real(nn.model(initial_states)), ps) 
            v = E_loc .* CUDA.ones(1, n_block_x)
            E_loc∂θ_j_b = back_j(v)[Flux.params(nn.model[k])[2]][block_out] # bias b*
            E_loc∂θ_j_W = back_j(v)[Flux.params(nn.model[k])[1]][block_out, block_in] # weight W* 
            E_loc∂θ .= [E_loc∂θ_j_b ; E_loc∂θ_j_W[:]]# transpose(cnoj(b))

            v = CUDA.ones(1, n_block_x)
            ∂θ_j_b = back_j(v)[Flux.params(nn.model[k])[2]][block_out] # bias b*
            ∂θ_j_W = back_j(v)[Flux.params(nn.model[k])[1]][block_out, block_in] # weight W* 
            ∂θ .= [conj(∂θ_j_b) ; conj(∂θ_j_W[:])]# transpose(cnoj(b))

            E_loc∂θ_j_b = nothing;
            E_loc∂θ_j_W = nothing;
            ∂θ_j_b = nothing;
            ∂θ_j_W = nothing;
            #end
        #end
    #end
     # transpose(cnoj(W))
end


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

function compute_nn_2nd_gpu(nn::FNN, initial_states::CuArray,block_out::Array{Int}, block_in::Array{Int})
    ∂lnψ_∂z = 1.0 
    nn.model[2].σ = nn.f_diff
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

########### --- Sampling --- ###########
########### --- Sampling --- ###########


mutable struct Sampling <: abstract_sampling
    n_thermals :: Int
    n_sweeps :: Int
    n_states :: Int #This means how many states we updates in one step(The states is a matrix, instead of an vector)
    counter :: Int
    state :: CuArray{Int,2}
end

mutable struct exact_sampling <: abstract_sampling
    counter :: Int
    n_states :: Int
    state :: CuArray{Int,2}
end


function thermalization!(quant_sys::T_model, model::Chain, sample::Sampling) where{T_model<:abstract_quantum_system, T1<:abstract_machine_learning}
    #sample.state = CuArray(rand(-1:2:1, quant_sys.n_sites, sample.n_states))
    #sample.state[1:Int(ising.n_sites/2),:] .= 1
    #sample.state[Int(ising.n_sites/2) + 1:end,:] .= -1
    #for m = 1 : ising.n_sites
    #    sample.state[m,:] .= (-1)^(m+1)
    #end
    #println("sample.state[:,1] = ",sample.state[:,1])

    for j = 1 : sample.n_thermals
        single_sweep!(quant_sys, model, sample)
    end
end



function gen_flip_list(flip_local::Array{Int,1}, n_sites::Int) # Generate a flip index
    N = length(flip_local)
    shift_num = collect(0: n_sites : (N-1) * n_sites)
    flip_global = flip_local .+ shift_num
    flip_list = [flip_local  flip_global]
    return flip_list
end



# markov chain
function single_sweep!(quant_sys::T_model, model::Chain, sample::Sampling) where{T_model<:abstract_quantum_system, T1<:abstract_machine_learning}
    L = quant_sys.n_sites
    for j = 1:L
        x_trial = copy(sample.state)
        #flip_local = Array(rand(1:L, N))
	    flip_local = zeros(Int, sample.n_states) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        x_trial[j,:] .= -x_trial[j,:]
        p = abs.(ratio_ψ(model,  x_trial, sample.state)).^2
        #println("p= ",p)
        r = CUDA.rand(1,sample.n_states) # generate random numbers 
        #println("r= ",r)
        sr = Array(r .< min.(1,p)) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = findall(x->x==1,sr[:]) # Find the indics of configurations which accept the new configurations
        site_ind_accept =  flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        sample.state[site_ind_accept] .= -sample.state[site_ind_accept]
    end
    #x_trial = nothing;
    #GC.gc()
end

#=======================#
#=======================#

########### --- main code: training parameters --- ###########
########### --- main code: training parameters --- ###########

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
end

mutable struct normal_optimization{ComplexF64} <: abstract_optimizer
end

#------------------------------------------------------------#

function optimization_params(opt_type::sr_optimization, n_params::Int)
    opt_params = Vector(undef, 4)
    opt_params[1] = 0 + 0im  # E_loc_avg
    opt_params[2] = CUDA.zeros(ComplexF64, n_params) # ∂θ_avg
    opt_params[3] = CUDA.zeros(ComplexF64, n_params) # E_loc∂θ_avg
    opt_params[4] = CUDA.zeros(ComplexF64, n_params, n_params) # ∂θ_mul_avg
    return opt_params
end


# this is for sampling method
function accum_opt_params!(opt_type::sr_optimization, opt_params_avg::Vector{Any}, 
    E_loc::CuArray{ComplexF64,2}, ∂θ :: CuArray{ComplexF64,2})
    opt_params_avg[1] += sum(E_loc)
    opt_params_avg[2] += sum(∂θ, dims = 2)
    opt_params_avg[3] += sum(E_loc .* conj(∂θ), dims = 2)
    opt_params_avg[4] .+= conj(∂θ) * transpose(∂θ)
end

#this is for exact sampling method
function accum_opt_params!(opt_type::sr_optimization, opt_params_avg::Vector{Any}, 
    E_loc::CuArray{ComplexF64,2}, ∂θ::CuArray{ComplexF64,2}, P_ψ::CuArray{Float64,2})
    opt_params_avg[1] += sum(P_ψ .* E_loc)
    opt_params_avg[2] += sum(P_ψ .* ∂θ, dims = 2)
    opt_params_avg[3] += sum(P_ψ .* E_loc .* conj(∂θ), dims = 2)
    opt_params_avg[4] .+= conj(P_ψ .* ∂θ) * transpose(∂θ)
end


function calculate_opt_params(opt_type::sr_optimization, opt_params_avg::Vector{Any})
    S = opt_params_avg[4] .- conj(opt_params_avg[2]) * transpose(opt_params_avg[2]) + opt_type.λ * I
    F = opt_params_avg[3] .- opt_params_avg[1] * conj(opt_params_avg[2])
    ∂θ = S \ F
    S = nothing;
    F = nothing;
    return ∂θ
    #return IterativeSolvers.cg( S, F )
end

#------------------------------------------------------------#
# normal optimiaztion with Monte Carlo method:
function optimization_params(opt_type::normal_optimization, n_params::Int)
    opt_params = Vector(undef, 3)
    opt_params[1] = 0 + 0im  # E_loc_avg
    opt_params[2] = CUDA.zeros(ComplexF64, n_params) # ∂θ_avg
    opt_params[3] = CUDA.zeros(ComplexF64, n_params) # E_loc∂θ_avg
    #opt_params[4] = CUDA.zeros(ComplexF64, n_params, n_params) # ∂θ_mul_avg
    return opt_params
end

#normal optimiaztion with exact sampling method
function accum_opt_params!(opt_type::normal_optimization, opt_params_avg::Vector{Any}, 
    E_loc::CuArray{ComplexF64,2}, ∂θ::CuArray{ComplexF64,2}, P_ψ::CuArray{Float64,2})
    opt_params_avg[1] += sum(P_ψ .* E_loc)
    opt_params_avg[2] += sum(P_ψ .* ∂θ, dims = 2)
    opt_params_avg[3] += sum(P_ψ .* E_loc .* conj(∂θ), dims = 2)
    #opt_params_avg[4] .+= conj(P_ψ .* ∂θ) * transpose(∂θ)
end

function accum_opt_params!(opt_type::normal_optimization, opt_params_avg::Vector{Any}, 
    E_loc::CuArray, ∂θ::CuArray{ComplexF64,2})
    opt_params_avg[1] += sum(E_loc)
    opt_params_avg[2] += sum(∂θ, dims = 2)
    opt_params_avg[3] += sum(E_loc .* conj(∂θ), dims = 2)
    #opt_params_avg[4] .+= conj(∂θ) * transpose(∂θ)
end

function calculate_opt_params(opt_type::normal_optimization, opt_params_avg::Vector{Any})
    #S = opt_params_avg[4] .- conj(opt_params_avg[2]) * transpose(opt_params_avg[2]) + opt_type.λ * I
    F = opt_params_avg[3] .- opt_params_avg[1] * conj(opt_params_avg[2])
    #∂θ = S \ F
    #S = nothing;
    #F = nothing;
    return F
    #return IterativeSolvers.cg( S, F )
end

# optimize parameters through markov chain
function optimize_params(::Type{T}, quant_sys::T_model, nn ::T1, sample::Sampling, 
    n_params::Int, block_in, block_out, layer_k, opt_type::T3, np::Int) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64, T1<:abstract_machine_learning, T3 <: abstract_optimizer}
    opt_params_avg = optimization_params(opt_type, n_params)
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)
    t_loc = 0.0
    t_der = 0.0
    t_acc = 0.0
    t_sample = @elapsed for j = 1 : sample.n_sweeps
        single_sweep!(quant_sys, nn.model, sample) # take samples by using Metropolis algorithm
        t_loc_j = @elapsed E_loc = calculate_E_local(quant_sys, nn.model, sample.state, sample.n_states, quant_sys.site_bond_y)
        t_der_j = @elapsed calculate_derivatives!(nn, sample.state, sample.n_states, ∂θ, np, block_in, block_out, layer_k)
        #t_der_j = @elapsed calculate_derivatives_any_hidden!(nn, sample.state, sample.n_states, ∂θ)

        #accum_opt_params!(opt_type, opt_params_avg, E_loc, ∂θ)
        t_acc_j = @elapsed accum_opt_params!(opt_type, opt_params_avg, E_loc, ∂θ)
        t_loc += t_loc_j
        t_der += t_der_j
        t_acc += t_acc_j
    end
    n_total = sample.n_sweeps * sample.n_states
    opt_params_avg ./= n_total
    t_opt = @elapsed ∂W = calculate_opt_params(opt_type, opt_params_avg)
    println("t_sample = $t_sample,  t_opt = $t_opt")
    println("t_loc = $t_loc,  t_der = $t_der,  t_acc = $t_acc")
    E_avg = opt_params_avg[1]
    opt_params_avg = nothing
    ∂θ = nothing
    return E_avg, ∂W
end


# train parameters throuh exact sampling
function optimize_params(::Type{T}, quant_sys::T_model, nn ::T1, sample::exact_sampling, 
    n_params::Int, block_in, block_out, layer_k, opt_type::T3, np::Int) #=
    =#   where{T_model<:abstract_quantum_system, T<:ComplexF64, T1<:abstract_machine_learning, T3 <: abstract_optimizer}
    
    opt_params_avg = optimization_params(opt_type, n_params)    
    E_loc = calculate_E_local(quant_sys, nn.model, sample.state, sample.n_states, quant_sys.site_bond_y)    
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.n_states)    
    calculate_derivatives!(nn, sample.state, sample.n_states, ∂θ, np, block_in, block_out, layer_k)    
    #calculate_derivatives_any_hidden!(nn, sample.state, sample.n_states, ∂θ)
    exact_ψ = exp.(nn.model(sample.state))    
    P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)   
    accum_opt_params!(opt_type, opt_params_avg, E_loc, ∂θ, P_ψ)    
    ∂W = calculate_opt_params(opt_type, opt_params_avg)    
    E_avg = opt_params_avg[1] 

    opt_params_avg = nothing
    ∂θ = nothing
    #GC.gc()
    return E_avg, ∂W
end

#=======================#
#=======================#

########### update type ###########
########### update type ###########

mutable struct gradient_descent <: abstract_update
    γ :: Float64
    final_γ :: Float64
    decay_rate::Float64
    decay_interval::Int
end

mutable struct adam_opt <: abstract_update
    γ :: Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m::Vector{Any}
    v::Vector{Any}
    t::Int
end

function adam_opt(γ, β1, β2, ϵ)
    adam_opt(γ, β1, β2, ϵ, [0.0], [0.0], 1)
end

#----------------------------------------------------

# update all parameters
function update_params!(grad_de::gradient_descent, nn::FNN, ∂W::CuArray{ComplexF64,1}, 
    block_in::String, block_out::String, layer_k::String)
    #print(∂W[1:10])
    for k = 1 : nn.n_layers - 1
        b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
        b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
        nn.model[k].b[:] .-= grad_de.γ * ∂W[b_ind_s : b_ind_f]
        W_ind_s = b_ind_f + 1
        W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
        nn.model[k].W[:,:] .-= grad_de.γ * reshape(∂W[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
        # println("layer $k : index of b at start = $b_ind_s")
        # println("layer $k : index of b at final = $b_ind_f")
        # println("layer $k : index of b at start = $W_ind_s")
        # println("layer $k : index of b at final = $W_ind_f")
        # println("==========================================")
    end
end

# update a part parameters in terms of block_in, block_out, layer_k
function update_params!(grad_de::gradient_descent, nn::FNN, ∂W::CuArray{ComplexF64,1}, 
    block_in::Array{Int}, block_out::Array{Int}, k::Int)
        nn.model[k].b[block_out] .-= grad_de.γ * ∂W[1 : length(block_out)]
        nn.model[k].W[block_out, block_in] .-= grad_de.γ * reshape(∂W[length(block_out) + 1 : end], length(block_out), length(block_in))
end

#----------------------------------------------------


function update_params!(adam::adam_opt,  nn::FNN, ∂W::CuArray{ComplexF64,1}, 
    block_in::String, block_out::String, layer_k::String)
	adam.m = adam.β1 .* adam.m .+ (1 - adam.β1) .* ∂W
    adam.v = adam.β2 .* adam.v .+ (1 - adam.β2) .* ∂W.^2
    m_hat = adam.m ./ (1 - adam.β1^adam.t)
    v_hat = adam.v ./ (1 - adam.β2^adam.t)
    dW = adam.γ .* m_hat ./ (sqrt.(v_hat) .+ adam.ϵ)
    for k = 1 : nn.n_layers - 1
        b_ind_s = sum(nn.layer_size[2:k]) + nn.layer_size[1:k-1]' * nn.layer_size[2:k]+ 1
        b_ind_f = b_ind_s - 1 + sum(nn.layer_size[k + 1]) 
        nn.model[k].b .-= dW[b_ind_s : b_ind_f]
        W_ind_s = b_ind_f + 1
        W_ind_f = W_ind_s - 1 + nn.layer_size[k] * nn.layer_size[k + 1]
        nn.model[k].W .-= reshape(dW[W_ind_s : W_ind_f], nn.layer_size[k + 1], nn.layer_size[k])
    end
end

#=======================#
#=======================#

########### --- main code: training parameters --- ###########
########### --- main code: training parameters --- ###########

function main_train_neuralnetwork(::Type{T}, quant_sys::T1, nn::FNN, sample::T4,
    update_type::abstract_update, opt_type::abstract_optimizer,
    n_epochs::Int, n_loops::Int, 
    block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
    file_name::String, np::Int) #=
    =# where {T<:Number, T1<:abstract_quantum_system, T4 <: abstract_sampling}
    # generate list of blocks of input and output
    block_in_list = [];
    block_out_list = [];

    # if upate all parameters
    if block_in_size == "all" && block_out_size == "all" && layer_type == "all"
        layer_list = ["all"]
        block_in_list = [["all"]]
        block_out_list = [["all"]]
    elseif layer_type == "layer"
        layer_list = collect(1 : nn.n_layers - 1)
        layer_list = [layer_list ; reverse(layer_list[2:end-1])]
        #layer_list = [layer_list ; reverse(layer_list[1:end])]
        # if update parameters in each block
        if block_in_size == "all" 
            for k = 1 : nn.n_layers - 1
                block_in_k = generate_blocks(nn.layer_size[k], nn.layer_size[k], 0)
                push!(block_in_list, block_in_k)
            end
        else 
            for k = 1 : nn.n_layers - 1
                block_in_k = generate_blocks(nn.layer_size[k], block_in_size[k], overlap_in[k])
                push!(block_in_list, block_in_k)
            end
        end
        
        if block_out_size == "all" 
            for k = 2 : nn.n_layers 
                block_out_k = generate_blocks(nn.layer_size[k], nn.layer_size[k], 0)
                push!(block_out_list, block_out_k)
            end
        else 
            for k = 2 : nn.n_layers 
                block_out_k = generate_blocks(nn.layer_size[k], block_out_size[k-1], overlap_out[k-1])
                push!(block_out_list, block_out_k)
            end
        end
    end

    if layer_type == "all"
        println("===> update all parameters !")
        n_params = sum(nn.layer_size[2:end]) + nn.layer_size[1:end-1]' * nn.layer_size[2:end]
        println("n_params = $n_params")
    elseif layer_type == "layer"
        for k = 1 : nn.n_layers - 1
            println("block of input $k is: 
            $(block_in_list[k][1][1]) ---> $(block_in_list[k][1][end]) 
            to $(block_in_list[k][end][1]) --> $(block_in_list[k][end][end])")
            println("length of block of input $k is: ", length(block_out_list[k]))
            println("block of output $k is: 
            $(block_out_list[k][1][1]) ---> $(block_out_list[k][1][end]) 
            to $(block_out_list[k][end][1]) --> $(block_out_list[k][end][end])")
            println("length of block of output $k is: ", length(block_out_list[k]))
        end
        println("layer_list = ", layer_list)
    end
    #println(block_out_list[1])
    
    n_params = 0
    if layer_type == "all"
        n_params = sum(nn.layer_size[2:end]) + nn.layer_size[1:end-1]' * nn.layer_size[2:end]
        if typeof(update_type) == adam_opt
            update_type.m = zeros(n_params)
            update_type.v = zeros(n_params)
        end
    end

    loop = 1
    # train parameters

    t_therm = @elapsed if typeof(sample) == Sampling
        println("warm up !!!")
        thermalization!(quant_sys, nn.model,  sample)
    end
    println("time for thermalization = $t_therm")
    #println("before optimization, the nn[1].W[1,1] = $(nn.model[1].W[1:2])")
    while  loop <= n_loops
        println("loop ++++++++++> $loop")
        for k = 1 : length(layer_list) # for loop of layers
            if layer_type == "all"
                l_k = 1
            else
                l_k = layer_list[k]
            end
            for out_m = 1 : length(block_out_list[l_k])
                block_out_m = block_out_list[l_k][out_m]

                for in_n = 1 : length(block_in_list[l_k])
                    block_in_n = block_in_list[l_k][in_n]
                    println("number of iterations = $(sample.counter)")
                    if typeof(update_type) == gradient_descent
                        if (sample.counter - 1)  % update_type.decay_interval == 0 && sample.counter != 1 && update_type.γ> update_type.final_γ + 1e-9
                            update_type.γ = update_type.γ * update_type.decay_rate
                            println(">>>>Learning rate = ", update_type.γ)
                        end
                    end
                    println("layer_k = $(layer_list[k])")
                    println("layer: $(layer_list[k]) ---> block_in  = $(block_in_n[1]) ---> $(block_in_n[end])")
                    println("layer: $(layer_list[k]) ---> block_out = $(block_out_m[1]) ---> $(block_out_m[end])")

                    if typeof(layer_list[k]) == Int
                        n_params = length(block_in_n) * length(block_out_m) + length(block_out_m)
                    end
                    #println(n_params)

                    for epoch = 1 : n_epochs
                        t_opt = @elapsed E_avg, ∂θ = optimize_params(T, quant_sys, nn,  sample, n_params, block_in_n, block_out_m, layer_list[k], opt_type, np)
                        if abs(E_avg) >= 1e+8 || isnan(E_avg)
                            error("Wrong ground energy: $E_avg !")
                        end
                        σx = 0
                        if quant_sys.n_sites <=14
                            σx = Measurement_ExactSampling(quant_sys, nn.model, 1)
                        else
                            σx = Measurement_Metropolis_Sampling(quant_sys, nn.model, sample, 1)
                        end
                        update_params!(update_type, nn, ∂θ[:], block_in_n, block_out_m, layer_list[k])
                        #println(" the nn[1].W[1:2] = $(nn.model[1].W[1:2])")
                        println("epoch = $epoch, E_avg = $E_avg;", "   σx = $σx")
                        io = open(file_name,"a+")
                        println(io, loop, "  $t_opt","      $E_avg")
                        close(io)
                        sample.counter += 1
                        if typeof(update_type) == adam_opt
                            update_type.t += 1
                        end
                        ∂θ = nothing
                        GC.gc()
                    end
                end

            end
        end
        println("==========================================")
        if mod(loop, 10) == 0
            GC.gc()
        end
        loop += 1
    end
        
end


#=======================#
#=======================#

########### --- main code: initialization --- ###########
########### --- main code: initialization --- ###########

function neural_network_initialization_run_gpu(::Type{T}, quant_sys_setting::Tuple, nn_setting::Tuple, sample_setting,
    update_setting::Tuple , opt_setting, np::Int) where{T<:Number}
    println("ready to run ! ")

    # initialize quantum system
    quant_sys_name = quant_sys_setting[1]
    Lattice = quant_sys_setting[2]
    Lx = Lattice[1]
    Ly = Lattice[2]
    n_sites = Lx*Ly
    if quant_sys_name == "Ising"
    
        J = quant_sys_setting[3]
        hx = quant_sys_setting[4]
        hz = quant_sys_setting[5]
        boundary_condition = quant_sys_setting[6]
        #correlation_num = quan_sys_setting[7]
        quant_sys = ising_model_initialization(J, hx, hz, Lx, Ly, boundary_condition)
    else
        error("There is no such a model: $quant_sys_name. The options of quantum system are: Ising.")
    end
    println("test")

    # initialize neural network
    if typeof(nn_setting[3]) == Array{Int, 1}
        nn_name = nn_setting[1]
        act_func_name = nn_setting[2]
        layer_size = nn_setting[3]
        n_layers = length(layer_size)
        random_seed_num = nn_setting[4]
        nn_distribution = nn_setting[5]
        distribution_simga = nn_setting[6]
        if nn_name == "FNN" 
            #neural_net = FNN_initialization(T, n_layers, layer_size,  random_seed_num, distribution_type, distribution_simga)
            act_f = activation_function(act_func_name)
            f_diff = activation_function_diff(act_func_name)
            neural_net = FNN_initialization(T, n_layers, layer_size,  act_f, f_diff,
            random_seed_num, distribution_type = nn_distribution, sigma = distribution_simga)
        else
            error("There is no such a neural network: $nn_name. The options of neural network are: FNN.")
        end
    else # which means we load a pre-trained neural network
        nn_name = nn_setting[1]
        act_func_name = nn_setting[2]
        random_seed_num = nn_setting[4]
        nn_distribution = nn_setting[5]
        distribution_simga = nn_setting[6]
        if nn_name == "FNN" 
            neural_net = deepcopy(nn_setting[3])
            layer_size = neural_net.layer_size
            n_layers = neural_net.n_layers
            for j = 1 : length(neural_net.model)
                neural_net.model[j].W = CuArray(neural_net.model[j].W)
                neural_net.model[j].b = CuArray(neural_net.model[j].b)
            end
        else
            error("There is no such a neural network: $nn_name. The options of neural network are: FNN.")
        end
    end

    println("W[1:2] = ",neural_net.model[1].W[1:2])

    

    # initialize sampling
    sample_name = sample_setting[1]
    if sample_name == "metropolis"
        n_thermals = sample_setting[2]
        n_sweeps = sample_setting[3]
        n_states = sample_setting[4]
        init_states = CuArray(rand(-1:2:1, n_sites, n_states))
        sample = Sampling(n_thermals, n_sweeps, n_states, 0, init_states)
        n_total = n_sweeps * n_states
    elseif sample_name == "exact"
        if n_sites > 16
            error("The number of sites is too large, we can not do exact sampling.")
        end
        n_exact_states = 2^(quant_sys.n_sites) # number of states for exact sampling
        exact_states = zeros(Int, quant_sys.n_sites, n_exact_states)

        for j = 0 : n_exact_states -1
            string_state = string(j, base = 2, pad = quant_sys.n_sites)
            for k = 1 : quant_sys.n_sites
                exact_states[k,j+1] = parse(Int,string_state[k])
            end
        end
        exact_states .= -2exact_states .+ 1
        n_states = n_exact_states
        sample = exact_sampling(0, n_states, CuArray(exact_states))
        n_total = n_states
        print("the sample = ")
        for k = 1 : 16
            print(sample.state[:,k])
        end

    else
        error("There is no such a sampling method: $sample_name. The options of sampling method are: metropolis, exact.")
    end

    if mod(sample.n_states, np) != 0 || np > sample.n_states
        error("The number of processors: $(sample.n_states / np) should be a divisor of the number of states.")
    end

    # initialize update parameters
    update_name = update_setting[1]
    n_epochs = update_setting[2]
    n_loops = update_setting[3]
    block_in_size = update_setting[4][1]
    block_out_size = update_setting[4][2]
    overlap_in = update_setting[5][1]
    overlap_out = update_setting[5][2]
    layer_type = update_setting[6]
    if update_name == "GD"
        γ = update_setting[7]
        final_γ = update_setting[8]
        decay_rate = update_setting[9]
        decay_interval = update_setting[10]
        update_type = gradient_descent(γ, final_γ, decay_rate, decay_interval)
    elseif update_name == "Adam"
        γ = update_setting[7]
        β1 = update_setting[8]
        β2 = update_setting[9]
        ϵ = update_setting[10]
        update_type = adam_opt(γ, β1, β2, ϵ)
    else
        error("There is no such a update method: $update_name. The options of update method are: GD and Adam.")
    end

    if layer_type == "all"
        if block_in_size != "all" || block_out_size != "all"
            error("The block_in_size and block_out_size should be: all when layer_type is all.")
        end
    end





    # initialize optimization
    opt_name = opt_setting[1]
    if opt_name == "SR"
        λ = opt_setting[2]
        opt_type = sr_optimization{ComplexF64}(λ)
    elseif opt_name == "normal"
        opt_type = normal_optimization{ComplexF64}()
        println("normal")
    else
        error("There is no such a optimization method: $opt_name. The options of optimization method are: SR and normal.")
    end

    output_file_name = "Ground_Energy_$(nn_name)_$(n_sites)_sites_$(n_total)_$(sample_name)_$(random_seed_num).txt"
    io = open(output_file_name, "w")
    io = open(output_file_name, "a+")

    # basic parameters in FileIO
    println(io, "Date:  ",(today()),"  ", Dates.Time(Dates.now()))
    println(io, "This is GPU version, and the number of threads is $np")
    println(io, "The ground energy of $quant_sys_name ; Lx = $Lx, Ly = $Ly, boundary_condition = $boundary_condition")
    if quant_sys_name == "Ising"
        println(io,"J = $J,   hx = $hx")
        println(io, "hz = $hz \n")
    end
    println(io, "==================================================")

    if nn_name == "FNN"
        println(io, "for FNN: layer structure is $(neural_net.layer_size), number of layers is $n_layers")
        println(io, "         activation function is $act_func_name")
        println(io, "         initialization of fnn: random seed number is $random_seed_num")
        println(io, "         distribution type is $nn_distribution, distribution sigma is $distribution_simga")
    end
    println(io, "==================================================")

    if sample_name == "metropolis"
        println(io, "the sampling method is Metropolis: the number of thermals is $n_thermals, the number of sweeps is $n_sweeps, the number of states is $n_states")
    elseif sample_name == "exact"
        println(io, "the sampling method is exact sampling: the total states is $n_states")
    end
    println(io, "==================================================")

    if update_name == "GD"
        println(io, "the update method is GD: the number of epochs is $n_epochs, the number of loops is $n_loops")
        println(io, "                         the initial learning rate is $γ, the final learning rate is $final_γ ")
        println(io, "                         the decay rate is $decay_rate, the decay interval is $decay_interval")
    end
    println(io, "==================================================")

    println(io, "the update method is: blocks of inputs =  $block_in_size, blocks of outputs = $block_out_size")
    println(io, "                      overlap of inputs = $overlap_in,    overlap of outputs = $overlap_out")
    println(io, "==================================================")

    if opt_name == "SR"
        println(io, "the optimization method is SR: the regularization parameter is $λ")
    end
    println(io, "==================================================")

    println(io, "Initial Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)

    # run code     
    println("Begin:")

    main_train_neuralnetwork(T,quant_sys, neural_net, sample, update_type, opt_type, 
    n_epochs, n_loops, 
    block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
    output_file_name, np)

    io = open(output_file_name, "a+")
    println(io, "The final learning rate is:  ", update_type.γ)
    println(io, "Final Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)
    println("Finish!")

    save_file_name = replace(output_file_name, ".txt" => "_gs.jld2")
    println(save_file_name)
    #println(neural_net.model[1])
    neural_net.model = neural_net.model |> cpu # doesn't change number type
    save(save_file_name, "nn_gs_cpu", neural_net)

end



#=======================#
#=======================#

########### --- measurment --- ###########
########### --- measurment --- ###########

function Measurement_ExactSampling(ising::ising_model, model::Chain,  site_n ::Int) 
    
    σ_avg = 0 
    σ_avg = σ_local_ExactSampling(ising.n_sites, model, site_n)
    #println("Exact Measurment") 
    return σ_avg
end

function σ_local_ExactSampling(L::Int, model::Chain,  i) 
    N_Exact = 2^L # number of states for exact sampling
    exact_state = zeros(Int, L, N_Exact)

    for j = 0 : N_Exact -1
        string_state = string(j, base = 2, pad = L)
        for k = 1 : L
            exact_state[k,j+1] = parse(Int , string_state[k])
        end
    end
    
    exact_state .= -2exact_state .+ 1
    exact_state = CuArray(exact_state)
    #println("exact state = ", exact_state)
    # exact_ψ = exp.(activation_func(act_func, fnn.b, fnn.W, exact_state, fnn.n_layers - 1))
    # P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)
    exact_lnψ = model(exact_state) 
    exact_lnψ_ratio = CUDA.@allowscalar exp.(exact_lnψ .- exact_lnψ[1])
    P_ψ = abs.(exact_lnψ_ratio).^2 / sum(abs.(exact_lnψ_ratio).^2)
    #println("P = $P_ψ")
    X = σ_flip(exact_state[:,:],i)

    #σ_loc = 0.0+0.0im
    new_states = X
    #println("new_state = $new_state")
    #σ_loc = Ratio_ψ(rbm, s_g.state, flip_global)
    #σ_loc = ψ_G(rbm, new_state, N_Exact)./ψ_G(rbm, exact_state[:,:], N_Exact)
    σ_loc = ratio_ψ(model, new_states, exact_state[:,:])
    #println(size(σ_loc))
    return sum(σ_loc .* P_ψ)
end


function σ_flip(x::CuArray{Int},i)
    X_new = copy(x);
    X_new[i,:] .= -X_new[i,:];
    return X_new;
end

function Measurement_Metropolis_Sampling(ising::ising_model, model::Chain,  sample::Sampling, site_n ::Int) 

    σ_avg = 0 
    X = σ_flip(sample.state[:,:] , site_n)
    new_states = X
    σ_avg = sum(ratio_ψ(model, new_states, sample.state[:,:])) / size(sample.state, 2)
    #println("Exact Measurment") 
    return σ_avg 
end
