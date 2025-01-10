# this code is for feedforward neural network model, quantum system and the acitivation functions 

using Flux
using CUDA
using Statistics
using Distributions 
using Random
using Dates
using JLD2
using Distributed
using LinearAlgebra
using Base.Iterators


abstract type abstract_quantum_system end # Define type of quantum systems: Ising model, Heisenberg models, ...

abstract type abstract_machine_learning{T <: Number} end # Define neural network models: Restricted Boltzmann Machine, Convetional Neural Network ...

abstract type abstract_activation_function end # Define type of activation functions: Relu, Leakey Relu, ...

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