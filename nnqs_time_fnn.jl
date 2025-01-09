#### This is for time evolution by trotter method and 
#### and we use neural network and Flux package to build up our quantum system

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

abstract type abstract_evolution end # Define type of evolution: trotter decomposition , taylor, RK2 or RK4, ...

########### ---Neural Network Models --- ###########
########### ---Neural Network Models --- ###########

# define feed forward neural network
mutable struct FNN{T} <: abstract_machine_learning{T} 
    model
    n_layers::Int
    layer_size::Array{Int,1}
    act_func  # activation function of first layer
    diff_func  # derivative of activation function of first layer
end

mutable struct FNN_time{T} <: abstract_machine_learning{T} 
    model
    n_layers::Int
    layer_size::Array{Int,1}
    act_func  # activation function of first layer
    diff_func  # derivative of activation function of first layer
end

mutable struct DenseC #neural network with complex parameters
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

function (a::DenseC)(x)
    #σ = a.σ
    a.σ(a.W * x .+ a.b)
end

Flux.@functor DenseC

function FNN_initialization(::Type{ComplexF64}, n_layers::Int, layer_size::Array{Int,1}, act_func::F, diff_func,
    random_seed_num::Int; 
    distribution_type::String = "gaussian", sigma::Float64 = 0.05)  where{F}
    println("sigma = ", sigma)
    if distribution_type == "gaussian"
        Random.seed!(random_seed_num)
        d = Normal(0, sigma)
    elseif distribution_type == "uniform"
        Random.seed!(random_seed_num)
        d = Uniform(-sigma, sigma)
        println("is uniform deitribution!")
    end
    layer = [];
    for k = 1 : n_layers - 1
        if k != n_layers - 1
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], act_func, d)]
        else
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], z -> z, d)]
        end
    end
    model = Chain(Tuple(layer))
    return FNN{ComplexF64}(model, n_layers, layer_size, act_func, diff_func)
end

function FNN_time_initialization(::Type{ComplexF64}, n_layers::Int, layer_size::Array{Int,1}, act_func::F, diff_func,
    random_seed_num::Int; 
    distribution_type::String = "gaussian", sigma::Float64 = 0.05)  where{F}
    println("sigma = ", sigma)
    if distribution_type == "gaussian"
        Random.seed!(random_seed_num)
        d = Normal(0, sigma)
    elseif distribution_type == "uniform"
        Random.seed!(random_seed_num)
        d = Uniform(-sigma, sigma)
        println("is uniform deitribution!")
    end
    layer = [];
    for k = 1 : n_layers - 1
        if k != n_layers - 1
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], act_func, d)]
        else
            layer = [layer; DenseC(layer_size[k] => layer_size[k+1], z -> z, d)]
        end
    end
    model = Chain(Tuple(layer))
    return FNN_time{ComplexF64}(model, n_layers, layer_size, act_func, diff_func)
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


# Val{1} means the closed boundary condition: the last site connects to the first site, and here we just consider the nearest-neighbor interaction
function configuration_initialization(x::Int , y::Int , :: Val{1} ) # Val{1} means the closed boundary condition
    if (x == 0 || y == 0) error("Wrong configuration setting: Number of site on X direction: $x,  Number of site on Y direction: $y") end
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

    if (x == 0 || y == 0) error("Wrong configuration setting: Number of site on X direction: $x,  Number of site on Y direction: $y") end
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

function get_coordination_number(N_sites::Int, bond_x :: Array{Int, 2}, bond_y :: Array{Int, 2})
    if size(bond_x , 1) == 0
        error("Wrong Configuration!")
    end

    L_z = Array(zeros(Int, N_sites, 2))

    # compute the coordination number for each site

    for j = 1 : N_sites
        C_n_j = count(c -> c == j, bond_x)
        L_z[j,1] = j
        L_z[j,2] = C_n_j
    end

    if size(bond_y) != 0
        for j = 1 : N_sites
            C_n_j = count(c -> c == j, bond_y)
            L_z[j,2] += C_n_j
        end
    end

    return L_z
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
    #return [block_list; block_list_reverse]
    if length(block_list) == 1
        return block_list
    else
        return [block_list; reverse(block_list)]
    end

end

#=======================#
#=======================#

########### ---  Time Evolution Type  --- ###########
########### ---  Time Evolution Type  --- ###########

mutable struct time_evolution
    initial_time::Float64
    final_time::Float64
    dt::Float64
    time::Array{Float64}
end
function time_evolution(initial_time::Real, final_time::Real, dt::Real)
    time = collect(initial_time : dt : final_time)
    return time_evolution(initial_time, final_time, dt, time)
end

mutable struct runge_kutta <: abstract_evolution
    order :: Int
    k :: Array{Int, 1}
end 

mutable struct taylor_expansion <: abstract_evolution 
end 

#=======================#
#=======================#

########### ---  Trotter Block  --- ###########
########### ---  Trotter Block  --- ###########

mutable struct Un
    U::CuArray{ComplexF64, 2}
    U_dag::CuArray{ComplexF64, 2}
    ind_U_col::CuArray{Int64, 1}
    ind_state::CuArray{Int64,2}
end
function Un(ising::ising_model, dt:: Float64, n::Int)  
    I2 = [1 0; 0 1]
    σx = [0 1; 1 0]
    σz = [1 0; 0 -1]
    σx_ij1 = kron(σx,I2)
    σx_ij2 = kron(I2,σx)

    σz_ij1 = kron(σz,I2)
    σz_ij2 = kron(I2,σz)

    σz_ij = kron(σz,σz)
    if n == 1
	    H_n = 0.5 * ising.hx * (2σx_ij1 + σx_ij2) .+ ising.J * σz_ij .+ 0.5 * ising.hz[n] * (2σz_ij1 + σz_ij2)
    elseif n == ising.n_sites - 1
	    H_n = 0.5 * ising.hx * (σx_ij1 + 2σx_ij2) .+ ising.J * σz_ij .+ 0.5 * ising.hz[n] * (σz_ij1 + 2σz_ij2)
    else
	    H_n =  0.5 * ising.hx * (σx_ij1 + σx_ij2) .+ ising.J * σz_ij .+ 0.5 * ising.hz[n] * (σz_ij1 + σz_ij2)
    end
    #H_n = 1.0* hx * kron(σx,I2) .+ J * σz_ij
    U = exp(-0.5*im*dt*H_n)
    U_dag = transpose(conj(U))
    #U = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    # index for U column 
    ind_U_col = [2,1]
    ind_state = [1 1 -1 -1;1 -1 1 -1]
    return Un(Array(U), Array(U_dag), Array(ind_U_col))
end

function Un_m(ising::ising_model, dt:: Float64, n::Int, N_H_bond::Int)  # for 1D case 
    if n + N_H_bond > ising.n_sites
        N_H_bond = ising.n_sites - n
        #error("The number of site exceeds!!!")
    end
    I2 = [1 0; 0 1]
    σx = [0 1; 1 0]
    σz = [1 0; 0 -1]
    Hx = zeros(Float64, 2^(N_H_bond+1), 2^(N_H_bond+1))
    Hz = zeros(Float64, 2^(N_H_bond+1), 2^(N_H_bond+1))
    Hzz = zeros(Float64, 2^(N_H_bond+1), 2^(N_H_bond+1))

    connect_bond_para = ones(Float64, N_H_bond + 1)
    if n == 1 
        connect_bond_para[end] = 0.5
    elseif n + N_H_bond == ising.n_sites 
        connect_bond_para[1] = 0.5
    else
        connect_bond_para[1] = 0.5
        connect_bond_para[end] = 0.5
    end

    #connect_bond_para = ones(Float64, N_H_bond + 1)
    
    for m = 1 :  N_H_bond + 1
        if m == 1
            Hx_m = σx 
            Hz_m = σz 
        else
            Hx_m = I2
            Hz_m = I2
        end

        for j = 2 : N_H_bond + 1
            if j == m
                Hx_m = kron(Hx_m, σx)
                Hz_m = kron(Hz_m, σz)
            else
                Hx_m = kron(Hx_m, I2)
                Hz_m = kron(Hz_m, I2)
            end
        end
        Hx = Hx .+ Hx_m * connect_bond_para[m]
        CUDA.@allowscalar Hz = Hz .+ Hz_m * connect_bond_para[m] * ising.hz[n+m-1]
    end

    for m = 1 : N_H_bond 
        if m == 1
            Hzz_m = kron(σz,σz)
        else
            Hzz_m = I2
            Hzz_m = I2
        end

        for j = 2 : N_H_bond 
            if j == m
                Hzz_m = kron(Hzz_m, σz,σz)
            else
                Hzz_m = kron(Hzz_m, I2)
            end
        end
        Hzz = Hzz .+ Hzz_m
    end

    H_n = ising.J * Hzz .+ ising.hx * Hx .+ Hz
    if n + N_H_bond == ising.n_sites
	    U = exp(-0.5 * im * dt * H_n)
    else
            U = exp(-0.5 * im * dt * H_n)
    end
    U_dag = transpose(conj(U))
    state2col = zeros(N_H_bond+1)
    c = 1

    for j = N_H_bond :-1 :0
        state2col[c] = 2^j
        c+=1
    end
    
    state2col = Int.(state2col)
    ind_state = zeros(Int,N_H_bond + 1, 2^(N_H_bond + 1))
    for j = 0: 2^(N_H_bond + 1) - 1
        string_state = string(j, base = 2, pad = N_H_bond + 1)
        for k = 1 : N_H_bond + 1
            ind_state[k,j+1] = parse(Int,string_state[k])
        end
    end
    ind_state .= -2ind_state .+ 1

    return Un(CuArray(U), CuArray(U_dag), CuArray(state2col), CuArray(ind_state))
    #return Un(U, U_dag, state2col, ind_state)

end


# This is unitary operator for 2D system, the default Hamiltonian bond is 1, so each local Hamiltonian inculdes 2 sites. 
function get_U(ising::ising_model, bond_m::Array{Int,1}, L_z::Array{Int,2}, dt::Float64)
    # generate Hx term
    #Jt = quench_parameters[1]
    #hxt = quench_parameters[2]
    #hzt = quench_parameters[3]
    I1 = Array([1 0 ; 0 1])
    sz = Array([1 0 ; 0 -1])
    sx = Array([0 1; 1 0])
    op = Dict("I" => I1, "σz" => sz, "σx" => sx)

    # generate Hx term
    #Hx_m_1 = generate_Ham_term_x_z( op["I"], op["σx"], bond_m[1], ising.N_sites) * (1 / L_z[bond_m[1], 2]) * ising.hx
    #Hx_m_2 = generate_Ham_term_x_z( op["I"], op["σx"], bond_m[2], ising.N_sites) * (1 / L_z[bond_m[2], 2]) * ising.hx
    Hx_m_1 = kron(sx, I1) * (1 / L_z[bond_m[1], 2]) * ising.hx
    Hx_m_2 = kron(I1, sx) * (1 / L_z[bond_m[1], 2]) * ising.hx

    # generate Hz term
    #Hz_m_1 = generate_Ham_term_x_z( op["I"], op["σz"], bond_m[1], ising.N_sites) * (1 / L_z[bond_m[1], 2]) * ising.hz[bond_m[1]]
    #Hz_m_2 = generate_Ham_term_x_z( op["I"], op["σz"], bond_m[2], ising.N_sites) * (1 / L_z[bond_m[2], 2]) * ising.hz[bond_m[2]]
    Hz_m_1 = kron(sz, I1) * (1 / L_z[bond_m[1], 2]) * ising.hz[bond_m[1]]
    Hz_m_2 = kron(I1, sz) * (1 / L_z[bond_m[2], 2]) * ising.hz[bond_m[2]]

    # generate Hzz term
    #H_zz_m = generate_Ham_term_zz( op["I"], op["σz"], bond_m, N_sites) * Jt
    H_zz_m = kron(sz, sz) * ising.J

    H_m = Hx_m_1 + Hx_m_2 + Hz_m_1 + Hz_m_2 + H_zz_m

    U = exp(-0.5 * im * dt * H_m)

    U_dag = transpose(conj(U))
    state2col = zeros(2)
    c = 1

    for j = 1 :-1 :0
        state2col[c] = 2^j
        c+=1
    end

    N_H_bond = 1

    state2col = Int.(state2col)
    ind_state = zeros(Int,N_H_bond + 1, 2^(N_H_bond + 1))
    for j = 0: 2^(N_H_bond + 1) - 1
        string_state = string(j, base = 2, pad = N_H_bond + 1)
        for k = 1 : N_H_bond + 1
            ind_state[k,j+1] = parse(Int,string_state[k])
        end
    end
    ind_state .= -2ind_state .+ 1

    return Un(CuArray(U), CuArray(U_dag), CuArray(state2col), CuArray(ind_state))
    #return Un(U, U_dag, state2col, ind_state)

end

function generate_Ham_term_x_z( I1 :: Array{Int,2}, s_op :: Array{Int,2}, n::Int, N_sites::Int)
    H_s = [1]
    for j = 1 : N_sites
        if j == n
            H_s = kron(H_s, s_op)
        else
            H_s = kron(H_s, I1)
        end
    end
    return H_s
end

function generate_Ham_term_zz( I1 :: Array{Int,2}, s_op :: Array{Int,2}, bond :: Array{Int, 1}, N_sites::Int)
    H_s = [1]
    for j = 1 : N_sites
        if j == bond[1] || j == bond[2]
            H_s = kron(H_s, s_op)
        else
            H_s = kron(H_s, I1)
        end
    end
    return H_s
end

function Rebuild_block_order(block_m :: Int, N :: Int) # N means the number of blocks
	max_d_r = N - block_m
	max_d_l = block_m - 1
        Order_ind = block_m;
	site_0 = block_m
	for j = 1 : 2 * max(max_d_l, max_d_r) + 1
		order_diret = (-1)^j
		add_order = collect(site_0 : order_diret : site_0 + order_diret * j)
                #println("add_order = " , add_order)
                popfirst!(add_order)
		Order_ind = [Order_ind; add_order]
		site_0 = site_0 + order_diret * j
	end
	Order_element = findall(x -> x>=1 && x<= N, Order_ind)
	Order_ind = Order_ind[Order_element]
	Order_ind = Order_ind[1 : end - (block_m - 1)]
	return Order_ind
end

#=======================#
#=======================#

########### --- activation function --- ###########
########### --- activation function --- ###########

function activation_function(activation::String)
    if activation == "lncosh"
        return f(z) = z .- z.^3/3 .+ z.^5 * 2/15 
    elseif activation == "lncosh_exact"
        return g(z) = log.(cosh.(z))
    else
        error("Wrong activation function")
    end
end

function activation_function_diff(activation::String)
    if activation == "lncosh"
        return f_diff(z) = 1 .- z.^2 .+ z.^4 * 2/3 
    elseif activation == "lncosh_exact"
        return g_diff(z) = tanh.(z)
    else
        error("Wrong activation function for derivative")
    end
end

#=======================#
#=======================#

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

###########  --- Sampling --- ###########
###########  --- Sampling --- ###########

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

########### --- main code: optimizing parameters --- ###########
########### --- main code: optimizing parameters --- ###########

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

#=======================#
#=======================#

########### --- measurment --- ########### 
########### --- measurment --- ###########

function Measurement_time_ExactSampling(ising::ising_model, model::Chain,  site_n ::Int) 
    
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


#=======================#
#=======================#

########### --- main code: training parameters --- ###########
########### --- main code: training parameters --- ###########

# main code for training
function main_train_time_trotter(::Type{T}, quant_sys::T1, nn::FNN, nn_t::FNN_time, sample::T4,
    update_type::abstract_update, opt_type,
    n_epochs::Int, n_loops::Int, n_ham_bond::Int,
    evo::time_evolution, tol::Float64, σ_i::Int, seed_num::Int,
    block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
    file_name::String) #=
    =# where {T<:Number, T1<:abstract_quantum_system, T3 <: abstract_activation_function, T4 <: abstract_sampling}
    
    ### generate list of blocks of input and output
    ### +++++++++++++++++++++++++++++++++++++++++++
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
        #layer_list = [1 , 2]
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
        n_params_adam = sum(nn.layer_size[2:end]) + nn.layer_size[1:end-1]' * nn.layer_size[2:end]
        if typeof(update_type) == adam_opt
            println("initialization m and v for adam:")
            update_type.m = zeros(n_params_adam)
            update_type.v = zeros(n_params_adam)
            println("initialize: $(update_type.m[1:5])")
        end
    elseif layer_type == "layer"
        for k = 1 : nn.n_layers - 1
            println("block of input $k is: 
            $(block_in_list[k][1][1]) ---> $(block_in_list[k][1][end]) 
            to $(block_in_list[k][end][1]) --> $(block_in_list[k][end][end])")
            println("length of block of input $k is: ", length(block_out_list[k]))
            println("block of output $k is: $(block_out_list[k][1][1]) ---> $(block_out_list[k][1][end]) to $(block_out_list[k][end][1]) --> $(block_out_list[k][end][end])")
            
            println("length of block of output $k is: ", length(block_out_list[k]))

        end
        println("layer_list = ", layer_list)
    end
    
    n_params = 0
    if layer_type == "all"
        n_params = sum(nn.layer_size[2:end]) + nn.layer_size[1:end-1]' * nn.layer_size[2:end]
        println("n_params = $n_params")
    end
    ### ++++++++++++++++++++++++++++++++++++++++++++++++++++
    ### ++++++++++++++++++++++++++++++++++++++++++++++++++++

    # generate hamiltonian indexing list
    #ham_ind = collect(1 : n_ham_bond : quant_sys.n_sites - n_ham_bond)
    ham_ind = collect(1 : n_ham_bond : quant_sys.n_sites)
    if ham_ind[end] == quant_sys.n_sites
        ham_ind = ham_ind[1:end-1]
    end
    ham_ind = [ham_ind ; reverse(ham_ind)]
    println("The hamiltonian indexing list is: ", ham_ind)

    # generate list of time steps
    tn = length(evo.time)

    # generate sample time
    sample_t = sampling_time(sample.state[:,:])
    # thermalization
    # if typeof(sample) == Sampling
    #     println("warm up the system !!! ============")
    #     thermalization!(quant_sys, nn, nn_t,  sample, sample_t)
    # end

    γ₀ = 0
    if typeof(update_type) == gradient_descent
        γ₀ = copy(update_type.γ)
    end
   

    #generate fnn_time
    

    overlap_t = 0

    println("Ready to run time evolution !!! ============")
    
    for t_ind = 2 : tn # time steps
        for m = ham_ind # hamiltonian bond index
            site_m = collect(m : min(m + n_ham_bond, quant_sys.n_sites))
            sample.counter = 0
            println("hamiltonian bond: ", site_m)
            
            # generate unitrary operator of hamiltonian
            Um = Un_m(quant_sys, evo.dt, m, n_ham_bond)

            # let ϕ(t+dt) = ψ(t)
            dt = Normal(0, 0.01)
            for c_k = 1 : nn.n_layers - 1
                nn_t.model[c_k].b .= nn.model[c_k].b * 1.0
                nn_t.model[c_k].W .= nn.model[c_k].W * 1.0
            end

            # for c_k = 1 : nn.n_layers - 1
            #     nn_t.model[c_k].b .= CuArray(rand(dt, size(nn_t.model[c_k].b)).+ im * rand(dt, size(nn_t.model[c_k].b)))
            #     nn_t.model[c_k].W .= CuArray(rand(dt, size(nn_t.model[c_k].W)).+ im * rand(dt, size(nn_t.model[c_k].W)))
            # end

            #println("Before updating:  b and W = ", nn_t.model[1].b[1], "    ", nn_t.model[1].W[1])

            if typeof(update_type) == gradient_descent
                update_type.γ = copy(γ₀)
            end
            println("learning rate = ",update_type.γ)

            if typeof(update_type) == adam_opt
                update_type.t = 1
                update_type.m .= 0.0 + 0.0im
                update_type.v .= 0.0 + 0.0im
            end

            if typeof(sample) == Sampling
                #println("warm up the system !!! ============")
                thermalization!(quant_sys, nn, nn_t,  sample, sample_t)
            end

            println("")
            loop = 1
            overlap_min = 1
            while loop <= n_loops  # training neural network
                println("loop ++++++++++> $loop")
                value_break = 0

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
                            println("layer: $(layer_list[k]) ==> block_in  = $(block_in_n[1]) ---> $(block_in_n[end])")
                            println("layer: $(layer_list[k]) ==> block_out = $(block_out_m[1]) ---> $(block_out_m[end])")

                            if typeof(layer_list[k]) == Int
                                n_params = length(block_in_n) * length(block_out_m) + length(block_out_m)
                            end

                            for epoch = 1 : n_epochs
                                # if mod(sample.counter, 1) == 0 && typeof(sample) == Sampling
                                #     single_sweep!(quant_sys, nn, sample)
                                #     single_sweep_time!(quant_sys, nn_t, sample_t)
                                # end
                                overlap, ∂θ = optimize_params(T, quant_sys, nn, nn_t, 
                                sample, sample_t, n_params, Um, site_m, 
                                block_in_n, block_out_m, layer_list[k], opt_type)
                                if abs(overlap) >= 1e+8 || isnan(overlap)
                                    error("Wrong ground energy: $overlap !")
                                end
                                update_params!(update_type, nn_t, ∂θ[:], block_in_n, block_out_m, layer_list[k])
                                println("epoch = $epoch, overlap = $overlap")
                                if typeof(update_type) == adam_opt
                                    update_type.t += 1
                                end
                                sample.counter += 1
                                overlap_t = abs(overlap)
                                overlap_min = min(overlap_min, overlap_t)
                                if (abs((overlap_t)) <= tol  && sample.counter >= 10) #|| s_g.counter >= 10 #&& m_c == length(Block_ind)
                                    value_break = 1	
                                    println("overlap_t = ",overlap_t,"   loop = ",loop)
                                    io_time = open(file_name,"a+")
                                    #println(io_time, m, " --->", loop, "      $overlap")
                                    println(io_time, m, " --->", sample.counter, "      $overlap_min         $overlap_t")
                                    close(io_time)
                                    break
                                end
                                
                                #println("t = ", update_type.t)

                            end # for loop: epoch

                            if value_break == 1
                                break
                            end

                        end # for loop: block_in
                        if value_break == 1
                            break
                        end
                    end # for loop: block_out
                    if value_break == 1
                        break
                    end
                end # for loop: layer

                if loop == n_loops
                    println("overlap_t = ",overlap_t,"   loop = ",loop)
                    io_time = open(file_name,"a+")
                    println(io_time, m, " --->", sample.counter, "      $overlap_min         $overlap_t")
                    close(io_time)
                end
                if value_break == 1
                    break
                end
                loop += 1
            end # while loop: training neural network
            for c_k = 1 : nn.n_layers - 1
                nn.model[c_k].b .= copy.(nn_t.model[c_k].b)
                nn.model[c_k].W .= copy.(nn_t.model[c_k].W)
            end
            sample.state .= copy.(sample_t.state_t)
        end # for loop: hamiltonian bond
        if typeof(sample) == Sampling
             n_total_save = sample.n_states * sample.n_sweeps
        elseif typeof(sample) == exact_sampling
            n_total_save = sample.n_states
        end
        # measurment 
        σx = 0
        if mod(t_ind - 1, 1) ==  0 || t_ind == 2
            if quant_sys.n_sites <= 14
                σx = Measurement_time_ExactSampling(quant_sys, nn.model,  σ_i)
            else
                σx = Measurement_Metropolis_Sampling(quant_sys, nn.model,  sample, σ_i)
            end
            io_time = open(file_name,"a+")
            println(io_time, evo.time[t_ind],": ", "  σx = ", σx)
            close(io_time)
            println("Time = ",evo.time[t_ind], "   Measurement= ", σx)
            hz1 = CUDA.@allowscalar quant_sys.hz[1]
            model_t = deepcopy(nn.model)
            model_t = model_t |> cpu
            save("./Time_Quench_FNN_$(quant_sys.n_sites)_sites_$(n_total_save)_tol_$(tol)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(quant_sys.hx)_hz_$(hz1).jld2", "model_t", model_t)
        end

    end # for loop: time steps

end # function


# t_VMC method for time evolution: the equation of motion is ∂W(t)/∂t = -im * S^{-1}(t) * F(t)
function main_rungle_kutta(::Type{T}, quant_sys::T1, nn::FNN, nn_t0::FNN_time, sample::T4, rk::rungle_kutta,
    update_type::abstract_update, opt_type::abstract_optimizer,
    evo::time_evolution, σ_i::Int, 
    file_name::String) #=
    =# where {T<:Number, T1<:abstract_quantum_system, T4 <: abstract_sampling}
    println("===> update all parameters !")
    layer_list = "all"
    block_in_list = "all"
    block_out_list = "all"
    n_params = sum(nn.layer_size[2:end]) + nn.layer_size[1:end-1]' * nn.layer_size[2:end]
    println("number of parameters is: $n_params")
    tn = length(evo.time)
    for  t_ind = 2 : length(evo.time)
        #println("Time = ",evo.time[t_ind])
        for c_k = 1 : nn.n_layers - 1
            nn_t0.model[c_k].b .= copy.(nn.model[c_k].b)
            nn_t0.model[c_k].W .= copy.(nn.model[c_k].W)
        end
        # thermalization
        for j = 1 : sample.n_thermals
            single_sweep!(quant_sys, nn, sample)
        end


        # nn_t0.b .= copy.(nn.b)
        # nn_t0.W .= copy.(nn.W)
        kn_save = Array{CuArray{ComplexF64}}(undef,rk.order)
        for n = 1 : rk.order
            overlap, kn = optimize_params(T, quant_sys, nn,  sample,  n_params, rk,  opt_type)
            println("E_avg = $overlap")
            if abs(overlap) >= 1e+8 || isnan(overlap)
                error("Wrong ground energy: $overlap !")
            end
            kn_save[n] = kn
            if n != 3
                update_params!(rk, nn, nn_t0, 0.5 * evo.dt * kn)
            elseif n == 3
                update_params!(rk, nn, nn_t0, evo.dt * kn)
            end
        end
        ∂θ = sum(rk.k .* kn_save) / 6 * evo.dt
        update_params!(rk, nn, nn_t0, ∂θ)
        

        if mod(t_ind - 1, 10) ==  0 || t_ind == 2
            if quant_sys.n_sites <= 14
                σx = Measurement_time_ExactSampling(quant_sys, nn.model,  σ_i)
            else
                σx = Measurement_Metropolis_Sampling(quant_sys, nn.model,  sample, σ_i)
            end
            io_time = open(file_name,"a+")
            println(io_time, evo.time[t_ind],": ", "  σx = ", σx)
            close(io_time)
            println("Time = ",evo.time[t_ind], "   Measurement= ", σx)
            #save("./Time_Quench_FNN_rk$(rk.order)_$(quant_sys.n_sites)_sites_$(sample.n_states)_tol_$(tol)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(quant_sys.hx)_hz_$(quant_sys.hz[1]).jld2", "fnn", nn, "act_func", typeof(act_func))
        end
        hz1 = CUDA.@allowscalar quant_sys.hz[1]
        n_total = sample.n_states * sample.n_sweeps
        if mod(t_ind - 1, 10) ==  0 || t_ind == 2
            model_t = deepcopy(nn.model)
            model_t = model_t |> cpu
            save("./Time_Quench_FNN_rk$(rk.order)_$(quant_sys.n_sites)_sites_$(n_total)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(quant_sys.hx)_hz_$(hz1).jld2", "fnn", model_t)
        end
    end # end for loop: time steps

end

#=======================#
#=======================#

########### --- Main Code: Initialization --- ###########
########### --- Main Code: Initialization --- ###########


function neural_network_initialization_time(quant_sys_setting::Tuple, nn_setting::Tuple, 
    sample_setting, update_setting::Tuple, opt_setting, time_setting::Tuple, tol::Float64, 
    evo_setting::Tuple, measurment_site::Int)
    T = ComplexF64
    # generate quantum system
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
    nn_name = nn_setting[1]
    act_func_name = nn_setting[2]
    nn_cpu = deepcopy(nn_setting[3])
    random_seed_num = nn_setting[4]
    nn_distribution = nn_setting[5]
    distribution_simga = nn_setting[6]
    if nn_name == "FNN"
        act_f = activation_function(act_func_name)
        diff_f = activation_function_diff(act_func_name)
        layer_size = nn_cpu.layer_size
        n_layers = nn_cpu.n_layers
        neural_net = FNN_initialization(T, n_layers, layer_size, act_f, diff_f, 
            random_seed_num, distribution_type = nn_distribution, sigma = distribution_simga)
        neural_net_time = FNN_time_initialization(T, n_layers, layer_size, act_f, diff_f, 
            random_seed_num, distribution_type = nn_distribution, sigma = distribution_simga)
        for k = 1 : n_layers -1
            neural_net.model[k].W = CuArray(nn_cpu.model[k].W)
            neural_net.model[k].b = CuArray(nn_cpu.model[k].b)
            neural_net_time.model[k].W = deepcopy(neural_net.model[k].W)
            neural_net_time.model[k].b = deepcopy(neural_net.model[k].b)
        end
    else
        error("There is no such a neural network: $nn_name. The options of neural network are: FNN.")
    end
    #println("model[2].W[1] = ", neural_net.model[2].W[1], "model[2].b[1] = ", neural_net.model[2].b[1])
    
    if nn_setting[7][1] == 1 # load time data
        model_t = nn_setting[7][2]
        for k = 1 : n_layers -1
            neural_net.model[k].W = CuArray(model_t[k].W)
            println("k = $k, size of W = ", size(neural_net.model[k].W))
            neural_net.model[k].b = CuArray(model_t[k].b)
            println("k = $k, size of b = ", size(neural_net.model[k].b))
            neural_net_time.model[k].W = deepcopy(neural_net.model[k].W)
            neural_net_time.model[k].b = deepcopy(neural_net.model[k].b)
        end
        println("loading time data: t = $(time_setting[1])")
    end
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
        n_exact_states = 2^(quant_sys.n_sites) # number of states for exact sampling
        exact_states = zeros(Int, quant_sys.n_sites, n_exact_states)

        for j = 0 : n_exact_states -1
            string_state = string(j, base = 2, pad = quant_sys.n_sites)
            for k = 1 : quant_sys.n_sites
                exact_states[k,j+1] = parse(Int,string_state[k])
            end
        end
        exact_states .= -2exact_states .+ 1
        exact_states = CuArray(exact_states)
        n_states = n_exact_states
        sample = exact_sampling(0, n_states, exact_states)
        n_total = n_states
    else
        error("There is no such a sampling method: $sample_name. The options of sampling method are: metropolis, exact.")
    end

    #load_nn_params = nn_params[1]

    # generate update parameters
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
        error("There is no such a update method: $update_name. The options of update method are: GD and adam.")
    end

    if layer_type == "all"
        if block_in_size != "all" || block_out_size != "all"
            error("The block_in_size and block_out_size should be: all when layer_type is all.")
        end
    end

    # generate time parameters
    t0 = time_setting[1]
    tf = time_setting[2]
    dt = time_setting[3]
    evo = time_evolution(t0, tf, dt)

    if evo_setting[1] == "trotter"
        n_ham_bond = evo_setting[2]
    elseif evo_setting[1] == "rk"
        if block_in_size != "all" || block_out_size != "all" ||layer_type != "all"
            error("The block_size_in and block_size_out should be: all when evolution type is rungle kutta method.")
        end
        order = evo_setting[2]
        rk_k_coff = evo_setting[3]
        rk = rungle_kutta(order, rk_k_coff)
    else 
        error("There are only two ways for time evolution: trotter, taylor and rk")
    end
    evo_mode = evo_setting[1]

    # initialize optimization
    opt_name = opt_setting[1]
    if opt_name == "SR"
        λ = opt_setting[2]
        α_r = opt_setting[3]
        opt_type = sr_optimization{ComplexF64}(λ, α_r)
    elseif opt_name == "minSR"
        λ = opt_setting[2]
        opt_type = min_sr_optimization(λ)
    elseif opt_name == "normal"
        opt_type = normal_optimization{ComplexF64}()
    elseif opt_name == "tvmc"
        opt_type = t_vmc{ComplexF64}()
    else
        error("There is no such a optimization method: $opt_name. The options of optimization method are: SR and tvmc.")
    end
    if evo_mode == "trotter"
        output_file_name = "time_evolution_$(nn_name)_$(n_sites)_sites_$(n_total)_$(sample_name)_tol_$(tol)_$(random_seed_num).txt"
    elseif evo_mode == "rk"
        output_file_name = "time_evolution_rk_$(nn_name)_$(n_sites)_sites_$(sample_name)_$(random_seed_num).txt"
    end
    io = open(output_file_name, "w")
    io = open(output_file_name, "a+")

    # basic parameters in FileIO
    println(io, "Date:  ",(today()),"  ", Dates.Time(Dates.now()))
    println(io, "Time evolution of $quant_sys_name ; Lx = $Lx, Ly = $Ly, boundary_condition = $boundary_condition")
    if quant_sys_name == "Ising"
        println(io,"J = $J,   hx = $hx")
        println(io, "hz = $hz")
    end
    println(io, "==================================================")

    println(io, "the initial time t0 = $t0, final time tf = $tf, time step dt = $dt")
    if evo_mode == "trotter"
        println(io,"the type of evolution is: $evo_mode, the number of bond is $n_ham_bond")
    elseif evo_mode == "rk"
        println(io,"the type of evolution is: $evo_mode, the order of rungle kutta is $order")
    end

    if nn_name == "FNN"
        println(io, "for FNN: layer structure is $(layer_size), number of layers is $n_layers")
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
    elseif update_name == "Adam"
        println(io, "the update method is Adam: the number of epochs is $n_epochs, the number of loops is $n_loops")
        println(io, "                         the initial learning rate is $γ")
    end
    println(io, "==================================================")

    println(io, "the update method is: blocks of inputs =  $block_in_size, blocks of outputs = $block_out_size")
    println(io, "                      overlap of inputs = $overlap_in,    overlap of outputs = $overlap_out")
    println(io, "==================================================")

    if opt_name == "SR"
        println(io, "the optimization method is $opt_name: the shift parameter λ is $λ, and the regularization parameter α_r = $α_r")
    end

    if opt_name == "minSR"
        println(io, "the optimization method is $opt_name: the shift parameter λ is $λ")
    end

    if opt_name == "normal"
        println(io, "the optimization method without SR, the normal way to update parameters")
    end

    println(io, "==================================================")
    
    # measurement 
    if quant_sys.n_sites <= 14
        σx = Measurement_time_ExactSampling(quant_sys, neural_net.model, measurment_site)
    else
        for j = 1 : sample.n_thermals
            single_sweep!(quant_sys, neural_net, sample)
        end
        σx = Measurement_Metropolis_Sampling(quant_sys, neural_net.model, sample, measurment_site)
    end
    println(io, "The measurement of σx at site $measurment_site is: $σx")
    println(io, "==================================================")

    println(io, "Initial Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)

    println( "The measurement of σx at site $measurment_site is: $σx")

    # run main code
    if evo_mode == "trotter"
        println("ready to run trotter")
        main_train_time_trotter(T, quant_sys, neural_net, neural_net_time, 
                            sample, update_type, opt_type, n_epochs, n_loops, n_ham_bond,
                            evo, tol, measurment_site, random_seed_num,
                            block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
                            output_file_name)
    elseif evo_mode == "rk"
        println("ready to run $evo_mode")
        main_rungle_kutta(T, quant_sys, neural_net, neural_net_time, 
                            sample, rk, update_type, opt_type, evo, measurment_site, 
                            output_file_name)
    end
    io = open(output_file_name, "a+")
    println(io, "Final Time:  ",(today()),"  ", Dates.Time(Dates.now()))  
    close(io)
end  