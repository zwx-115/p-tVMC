using Base: Float64
using LinearAlgebra
using Statistics
using Distributions 
using Random
using CUDA
using Dates
using JLD
using JLD2

abstract type Abstract_Quantum_System end # Define type of quantum systems: Ising model, Heisenberg models

abstract type Abstract_Learning_Machine{T <: Number} end # Define neural network models: Restricted Boltzmann Machine, Convetional Neural Network ...

abstract type Abstract_Sampling end # Define sampling methods: Monte Carlo, Metropolis Hsating ...

########### ---Neural Network Models --- ###########
########### ---Neural Network Models --- ###########

export RBM, RBM_Initialization
mutable struct RBM{T} <: Abstract_Learning_Machine{T} 
    a::CuArray{T, 1}
    b::CuArray{T, 1}
    W::CuArray{T, 2}
    _nv::Int
    _nh::Int
    α::Float64
end

function RBM_Initialization(::Type{Float64}, nv::Int, nh::Int, α::Float64, random_seed_num::Int; sigma = 0.001)   # Initializing data of RBM if float number
    Random.seed!(random_seed_num)
    d = Normal(0, sigma)
    a = rand( d, nv)
    b = rand( d, nh)
    W = rand( d, nh, nv)
    
    return RBM{Float64}(a,  W, nv, nh, α)
    #return RBM{Float64}(a, b, W, nv, nh, α)
end

function RBM_Initialization(::Type{ComplexF64}, nv::Int, nh::Int, α::Float64, random_seed_num::Int; sigma = 0.001)   # Initializing data of RBM if complex number
    Random.seed!(random_seed_num)
    d = Normal(0, sigma)
    a = rand(d, nv)  .+ im * rand(d, nv)
    b = rand(d, nh)  .+ im * rand(d, nh)
    W = rand(d, nh, nv) .+ im * rand(d, nh, nv)

    #return RBM{ComplexF64}(a,  W, nv, nh, α)
    return RBM{ComplexF64}(a, b, W, nv, nh, α)
end

#=======================================#
#=======================================#


########### --- Quantum Systems --- ###########
########### --- Quantum Systems --- ###########

export Ising_Model, Configuration_Initialization, Ising_Model_Initialization
export Heisenberg_Model, Heisenberg_Model_Initialization

mutable struct Ising_Model <: Abstract_Quantum_System
    J :: Float64
    hx :: Float64
    hz :: CuArray{Float64,1}
    n_sites :: Int
    site_bond_x :: CuArray{Int,2}
    site_bond_y :: CuArray{Int}
end
function Ising_Model_Initialization(J::Float64, hx::Float64, hz::Array{Float64, 1}, n_sites_x::Int, n_sites_y::Int, BC::String) #BC means Boundary conditions
    hz = CuArray(hz)
    n_sites = n_sites_x * n_sites_y

    if BC == "open" #open boundary condition
        x = 0
    elseif BC == "closed" #closed boundary condition
        x = 1
    else 
        error("No Such boundary Condition: $BC.  The option of boundary condition are: closed and open")
    end
    site_bond_x, site_bond_y = Configuration_Initialization(n_sites_x, n_sites_y, Val(x))
    return Ising_Model(J, hx, hz, n_sites, site_bond_x, site_bond_y)

end

mutable struct Heisenberg_Model <: Abstract_Quantum_System
	J :: Float64
	n_sites :: Int    
	site_bond_x :: CuArray{Int,2}
	site_bond_y :: CuArray{Int}
end

function Heisenberg_Model_Initialization(J::Float64, n_sites_x :: Int, n_sites_y :: Int, BC::String)#BC means Boundary conditions
	n_sites = n_sites_x * n_sites_y
	if BC == "open" #open boundary condition
		x = 0
	elseif BC == "closed" #closed boundary condition
		x = 1
	else
		error("No Such boundary Condition: $BC.  The option of boundary condition are: closed and open")
	end
	site_bond_x, site_bond_y = Configuration_Initialization(n_sites_x, n_sites_y, Val(x))
	return Heisenberg_Model(J, n_sites, site_bond_x, site_bond_y)
end


# Val{1} means the closed boundary condition: the last site connects to the first site, and here we just consider the nearest-neighbor interaction
function Configuration_Initialization(x::Int , y::Int , :: Val{1} ) # Val{1} means the closed boundary condition
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
    return CuArray(Site_Bond_x), CuArray(Site_Bond_y)
end

# Val{0} means the open boundary condition: the last site cannot connect to the first site, and here we just consider the nearest-neighbor interaction
function Configuration_Initialization(x::Int , y::Int , :: Val{0} ) # Val{0} means the open boundary condition

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

    return CuArray(Site_Bond_x), CuArray(Site_Bond_y)
end


function generate_block_2D(Lx :: T, Ly :: T, block_size:: Array{T,1}) where{T<:Int}
	block_size_x = block_size[1]
	block_size_y = block_size[2]
	Block_ind = []
	if block_size_x != block_size_y
		error("For the 2D system, the blocks size in x and y direction should be equal! ")
	end
	if mod(block_size_x , 2) == 0
	N_blocks_x = ceil(Int,Lx/block_size_x)
	N_blocks_y = ceil(Int,Ly/block_size_y)
#	Block_ind = []
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
			push!(Block_ind, CuArray(Block_xy))
		end
	end
	else
	N_blocks_x = Int(ceil(Lx/block_size_x))
	N_blocks_y = Int(ceil(Ly/block_size_y))
#	Block_ind = []
	for j = 1 : 2 * N_blocks_y
		block_size_overlap_y = Int((block_size_y - 1)/2)
		block_ind_y = collect(1 : block_size_y) .+ block_size_overlap_y * (j - 1)
		block_ind_y = (block_ind_y .-1).* Lx .+1
		for k = 1 : 2 * N_blocks_x 
			Block_xy = Array{Int}(undef,0)
			block_size_overlap_x = Int((block_size_x - 1)/2)
			for k_y = 1 : length(block_ind_y)
				Block_local_xy = block_ind_y[k_y] .- 1 .+ collect(1 : block_size_x) .+ block_size_overlap_x * (k - 1)
				Block_xy = [Block_xy; Block_local_xy]
			end
			push!(Block_ind, CuArray(Block_xy))
		end
	end
	end
	return Block_ind
end

function generate_block_2D(quan_sys :: Abstract_Quantum_System) # The case where the block size in x = 2, and in y = 1 
	Block_ind = []
	for j = 1 : length(quan_sys.site_bond_x[:,1])
		push!(Block_ind, quan_sys.site_bond_x[j,:] )
	end

	for j = 1 : length(quan_sys.site_bond_y[:,1])
		push!(Block_ind, quan_sys.site_bond_y[j,:] )
	end
	return Block_ind

end	



#=======================================#
#=======================================#

########### --- Local Energy: E_loc --- ###########
########### --- Local Energy: E_loc --- ###########

export Ratio_ψ, gen_flip_list, E_local_G

function ψ_G(rbm_g::RBM, x::CuArray{Int64}, N::Int) # calculate wave function ψ using GPU
    aT = reshape(rbm_g.a,1,length(rbm_g.a))
    #b = rbm_g.b
    #b = repeat(rbm_g.b,1,N)
    #W = rbm_g.W
    #println("size of C",size(W * x))
    C = rbm_g.W * x .+ rbm_g.b
    
    return CUDA.reduce(*, [exp.(aT * x);  (exp.(C).+exp.(-C))],dims =1)
end

function Ratio_ψ(rbm_g::RBM, x::CuArray{Int64, 2}, flip_list::CuArray{Int64,2}) # calculate wave function ψ on GPU
    Δa = -2*rbm_g.a[flip_list[:,1]] .* x[flip_list[:,2]]
    C = rbm_g.W * x .+ rbm_g.b
    C_new  = C .- 2*rbm_g.W[:,flip_list[:,1]] .* transpose(x[flip_list[:,2]])
    return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
end

function Ratio_ψ(rbm_g::RBM, x::CuArray{Int64, 2}, flip_list_1::CuArray{Int64,2}, flip_list_2::CuArray{Int64,2}) # calculate wave function ψ on GPU
	Δa = -2*rbm_g.a[flip_list_1[:,1]] .* x[flip_list_1[:,2]] .- 2*rbm_g.a[flip_list_2[:,1]] .* x[flip_list_2[:,2]]
	C = rbm_g.W * x .+ rbm_g.b
	C_new  = C .- 2*rbm_g.W[:,flip_list_1[:,1]] .* transpose(x[flip_list_1[:,2]]) .- 2*rbm_g.W[:,flip_list_2[:,1]] .* transpose(x[flip_list_2[:,2]])
	return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
end

function gen_flip_list(flip_local::CuArray{Int,1}, n_sites::Int) # Generate a flip index
    N = length(flip_local)
    shift_num = CuArray(collect(0: n_sites : (N-1) * n_sites))
    flip_global = flip_local .+ shift_num
    flip_list = [flip_local  flip_global]
    return flip_list
end

function E_local_G(ising::Ising_Model, rbm_g::RBM, initial_state::CuArray{Int, 2} , N::Int, y::CuArray{Int,1}) # calaulate E_loc for Ising model under 1D case
    E_loc = CuArray(zeros(ComplexF64,1,N))
    E_loc .+= ising.hz' * (initial_state.*1.0) # diagnoal part
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_x[:,1], :] .* initial_state[ising.site_bond_x[:,2],:],dims=1) # diagnoal part for bond x
    flip_local = CuArray(zeros(Int, N))
    flip_list_0 = gen_flip_list(flip_local, ising.n_sites)
    # calculate the off-diagonal part
    for j = 1 : ising.n_sites
        flip_list_j = flip_list_0 .+ j
        E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j) .* ising.hx
    end
    return E_loc
end

function E_local_G(ising::Ising_Model, rbm_g::RBM, initial_state::CuArray{Int, 2} , N::Int, y::CuArray{Int,2}) # calaulate E_loc for Ising model under 2D case
    E_loc = CuArray(zeros(ComplexF64,1,N))
    E_loc .+= ising.hz' * (initial_state.*1.0) # diagnoal part
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_x[:,1], :] .* initial_state[ising.site_bond_x[:,2],:],dims=1) # diagnoal part for bond x
    E_loc .+= ising.J .* sum(initial_state[ising.site_bond_y[:,1], :] .* initial_state[ising.site_bond_y[:,2],:],dims=1) # diagnoal part for bond y

    flip_local = CuArray(zeros(Int, N))
    flip_list_0 = gen_flip_list(flip_local, ising.n_sites)
    # calculate the off-diagonal part
    for j = 1 : ising.n_sites
        flip_list_j = flip_list_0 .+ j
        E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j) .* ising.hx
    end
    return E_loc
end

# Heisenberg Model: Compute E_loc
function E_local_G(heisenberg::Heisenberg_Model, rbm_g::RBM, initial_state::CuArray{Int, 2} , N::Int, y::CuArray{Int,1}) # calaulate E_loc for Heisenberg model under 1D case
	E_loc = CuArray(zeros(ComplexF64,1,N))
	Szz = initial_state[heisenberg.site_bond_x[:,1], :] .* initial_state[heisenberg.site_bond_x[:,2],:]
	E_loc .+= 0.25 * heisenberg.J .* sum(Szz , dims=1) # diagnoal part for bond x
	Szz_coff = -0.5 * (Szz .- 1) # if both spins are up or down, cofficient is 0, otherwise cofficient is 1 
	flip_local = CuArray(zeros(Int, N))
	flip_list_0 = gen_flip_list(flip_local, heisenberg.n_sites)
	# calculate the off-diagonal part
	for j = 1 : length(heisenberg.site_bond_x[:,1])
		flip_list_j = flip_list_0 .+ heisenberg.site_bond_x[j,1] # flip the first spin
		flip_list_bond = flip_list_0 .+ heisenberg.site_bond_x[j,2] # flip the second spin
		E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j, flip_list_bond) .* transpose(Szz_coff[j,:]) * 0.5 * heisenberg.J 
	end
	return E_loc

end

function E_local_G(heisenberg::Heisenberg_Model, rbm_g::RBM, initial_state::CuArray{Int, 2} , N::Int, y::CuArray{Int,2}) # calaulate E_loc for Heisenberg model under 2D case
	E_loc = CuArray(zeros(ComplexF64,1,N))
	Szz_x = initial_state[heisenberg.site_bond_x[:,1], :] .* initial_state[heisenberg.site_bond_x[:,2],:]
	Szz_y = initial_state[heisenberg.site_bond_y[:,1], :] .* initial_state[heisenberg.site_bond_y[:,2],:]
	E_loc .+= 0.25 * heisenberg.J .* sum(Szz_x , dims=1) # diagnoal part for bond x
	E_loc .+= 0.25 * heisenberg.J .* sum(Szz_y , dims=1) # diagnoal part for bond y
	Szz_x_coff = -0.5 * (Szz_x .- 1) # if both spins are up or down, cofficient is 0, otherwise cofficient is 1 
	Szz_y_coff = -0.5 * (Szz_y .- 1) # if both spins are up or down, cofficient is 0, otherwise cofficient is 1
	flip_local = CuArray(zeros(Int,N))
	flip_list_0 = gen_flip_list(flip_local, heisenberg.n_sites)
	# calculate the off-diagonal part
	for j = 1 : length(heisenberg.site_bond_x[:,1])
		flip_list_j = flip_list_0 .+ heisenberg.site_bond_x[j,1] # flip the first spin
		flip_list_bond = flip_list_0 .+ heisenberg.site_bond_x[j,2] # flip the second spin
		E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j, flip_list_bond) .* transpose(Szz_x_coff[j,:]) * 0.5 * heisenberg.J
	end

	for j = 1 : length(heisenberg.site_bond_y[:,1])
		flip_list_j = flip_list_0 .+ heisenberg.site_bond_y[j,1] # flip the first spin
		flip_list_bond = flip_list_0 .+ heisenberg.site_bond_y[j,2] # flip the second spin
		E_loc .+= Ratio_ψ(rbm_g, initial_state, flip_list_j, flip_list_bond) .* transpose(Szz_y_coff[j,:]) * 0.5 * heisenberg.J
	end
	return E_loc
end

#=======================================#
#=======================================#

########### --- Dervatives: ∂θ and construction of S --- ###########
########### --- Dervatives: ∂θ and construction of S --- ###########

export measure_G

# nm::String means that all parameters are updated
#function measure_G(ising::Ising_Model, rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T,2}, N::Int, nm::String) where{T<:Number}
function measure_G(quant_sys::T1, rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T2,2}, N::Int, nm::String) where{T1 <: Abstract_Quantum_System, T2<:Number}
    E_loc = E_local_G(quant_sys, rbm_g, state, N, quant_sys.site_bond_y)
    ∂θ = Compute_derivatives_G(rbm_g, state, N)
    E_loc, ∂θ, E_loc .* conj(∂θ) #∂θ just represents one portion of the whole derivatives of parameters.
end

# nm::Int means that a and b are updated
function measure_G(quant_sys::T1, rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T2,2}, N::Int, nm::Int) where{T1 <: Abstract_Quantum_System, T2<:Number}
    E_loc = E_local_G(quant_sys, rbm_g, state, N, quant_sys.site_bond_y)
    ∂θ = Compute_derivatives_G(rbm_g, state, ∂θ, N, nm)#(rbm_g::RBM_G, state::CuMatrix{Int}, ∂W, nm)
    
    E_loc, ∂θ, E_loc .* conj(∂θ) #∂θ just represents one portion of the whole derivatives of parameters.
end

# nm::CuArray{Int,1} means that a and W are updated
function measure_G(quant_sys::T1, rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T2,2}, N::Int, nm::CuArray{Int,1}) where{T1  <: Abstract_Quantum_System, T2 <: Number}
    E_loc = E_local_G(quant_sys	, rbm_g, state, N, quant_sys.site_bond_y)
    ∂θ = Compute_derivatives_G(rbm_g, state, ∂θ, N, nm)#(rbm_g::RBM_G, state::CuMatrix{Int}, ∂W, nm)
    
    E_loc, ∂θ, E_loc .* conj(∂θ) #∂θ just represents one portion of the whole derivatives of parameters.
end

# --------------------------------------------
# --------------------------------------------

function Compute_derivatives_G(rbm_g::RBM, state::CuArray{Int, 2}, N::Int)  # All parameters: compute ∂a , ∂b and ∂W 
    ∂a = state 
    #println("∂a= ",typeof(∂a))
    C  = rbm_g.W * state .+ rbm_g.b
    ∂b = (exp.(C).-exp.(-C))./(exp.(C).+exp.(-C))
    #println("-----",typeof(∂b))
    #∂b = tanh.(C)
    ∂W = CuArray(zeros(ComplexF64,size(∂b,1)*size(∂a,1),N))
    CUDA.@sync begin
        @cuda blocks = N Compute_∂W!(∂W, ∂a, ∂b)
    end
    #return [∂a; ∂W]
    return [∂a; ∂b; ∂W]
end

function Compute_derivatives_G(rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T,2}, N::Int, nm::Int) where{T<:Number} # Compute ∂a and ∂b
    ∂a = state 
    C  = rbm_g.b .+ rbm_g.W * state
    ∂b = (exp.(C).-exp.(-C))./(exp.(C).+exp.(-C))
    ∂θ .= [∂a; ∂b]
    return ∂θ
end

export Compute_derivatives_G
function Compute_derivatives_G(rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T,2}, N::Int, nm::CuArray{Int, 1}) where{T<:Number}  # Compute a part of ∂a , all  ∂b and  ∂W
    ∂a_nm = state[nm,:]
    #println("∂a= ",typeof(∂a))
    C  = rbm_g.W * state .+ rbm_g.b
    ∂b = (exp.(C).-exp.(-C))./(exp.(C).+exp.(-C))
    #println("-----",typeof(∂b))
    #∂b = tanh.(C)
    ∂W = CuArray(zeros(T, size(∂b, 1)*size(∂a_nm, 1), N)) #∂θ is a column vector
    CUDA.@sync begin
        @cuda blocks = 128 threads = 512 Compute_∂W_nm!(∂W, ∂a_nm, ∂b, N)
    end
    ∂θ .= [∂a_nm; ∂b;  ∂W]
    #∂θ .= [∂a_nm;  ∂W]
    #∂θ .= ∂W
    return ∂θ
end


function Compute_∂W!(∂W, ∂a, ∂b)# Compute ∂W when the system is relatively small.
    ind_x = blockIdx().x
    n_b = size(∂b,1)
    n_a = size(∂a,1)
    for i = 1 : n_b
        for j = 1 : n_a
            ∂W[i+n_b*(j-1),ind_x] = ∂b[i,ind_x] *∂a[j,ind_x]
        end
    end

    return nothing
end

function Compute_∂W_nm!(∂θ, ∂a_nm, ∂b, N)# Compute a part of ∂W .
    ind_th = threadIdx().x   # thread means different elements in W
    ind_b = blockIdx().x     # Block means different configurations
    dim_th = blockDim().x    #number of threads
    dim_b = gridDim().x      #number of blocks
    n_b = size(∂b,1)
    n_a = size(∂a_nm,1)
    for k = ind_b : dim_b : N
        for i = ind_th : dim_th : n_b
            for j = 1 : n_a
                ∂θ[i+n_b*(j-1),k] = ∂b[i,k] * ∂a_nm[j,k]
            end
        end
    end
    return nothing
end


function  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, N_sampling)
    ind_x = blockIdx().x #Index of block, corresponds to the index of column
    ind_th_x = threadIdx().x #Index of threads which corresponds to the index of row 
    stride = blockDim().x #number of threads
    stride_b = gridDim().x #number of blocks
    n_∂θ = size(∂θ_nm,1) # number of parameters
    #for j = ind_th_x:stride:n_∂θ
    for i = ind_th_x:stride:n_∂θ
        for j = ind_x:stride_b:n_∂θ
            for n = 1:N_sampling
                #S_kk[ind_x,j] += conj(∂θ_nm[ind_x,n]) * ∂θ_nm[j,n]
                S_kk[i,j] += conj(∂θ_nm[i,n]) * ∂θ_nm[j,n]
            end
        end
    end
    return nothing
end

#=======================================#
#=======================================#

########### --- Sampling --- ###########
########### --- Sampling --- ###########

export Sampling, Thermalization!, Single_Sweep!
mutable struct Sampling <: Abstract_Sampling
    n_thermals :: Int
    n_sweeps :: Int
    N_states :: Int #This means how many states we updates in one step(The states is a matrix, instead of an vector)
    counter :: Int
    state :: CuArray{Int,2}
end

mutable struct exact_sampling <: Abstract_Sampling
    counter :: Int
    n_states :: Int
    state :: CuArray{Int,2}
end

function Thermalization!(ising::Ising_Model, rbm ::T, sample::Sampling) where{T<:Abstract_Learning_Machine}
    sample.state = CuArray(rand(-1:2:1, ising.n_sites, sample.N_states))
    #sample.state[1:Int(ising.n_sites/2),:] .= 1
    #sample.state[Int(ising.n_sites/2) + 1:end,:] .= -1
    #for m = 1 : ising.n_sites
    #    sample.state[m,:] .= (-1)^(m+1)
    #end
    println("sample.state[:,1] = ",sample.state[:,1])

    for j = 1 : sample.n_thermals
        Single_Sweep!(ising, rbm, sample)
    end
end

function Thermalization!(heisenberg::Heisenberg_Model, rbm ::T, sample::Sampling) where{T<:Abstract_Learning_Machine}
    sample.state = CuArray(rand(-1:2:1, heisenberg.n_sites, sample.N_states))
    sample.state[1:Int(heisenberg.n_sites/2),:] .= 1
    sample.state[Int(heisenberg.n_sites/2)+1 : end , :] .= -1
   # println("====>>>", sample.state)
    for j = 1 : sample.n_thermals
        Single_Sweep!(heisenberg, rbm, sample)
    end
end

function Single_Sweep!(quant_sys::Ising_Model, rbm ::T2, sample::Sampling) where{ T2<:Abstract_Learning_Machine}
    L = quant_sys.n_sites
    state_new = sample.state
    N = sample.N_states # Number of chains for each step
    for j = 1:L
        x_trial = copy(sample.state)
        #flip_local = CuArray(rand(1:L, N))
	    flip_local = CuArray(zeros(Int,N)) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        p = abs.(Ratio_ψ(rbm, sample.state, flip_global)).^2
        #println("p= ",p)
        r = CUDA.rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = CuArray(findall(x->x==1,sr[:])) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        sample.state[site_ind_accept] .= -sample.state[site_ind_accept]
    end
end

function Single_Sweep!(quant_sys::Heisenberg_Model, rbm ::T2, sample::Sampling) where{ T2<:Abstract_Learning_Machine}
    L = quant_sys.n_sites
    state_new = sample.state
    N = sample.N_states # Number of chains for each step
    for j = 1:L
        #x_trial = copy(sample.state)
	flip_local_1 = CuArray(zeros(Int, N)) .+ j #CuArray(rand(1:L, N))
	flip_local_1 = CuArray(rand(1:L, N))
	#println("flip_local = ", flip_local_1)
	flip_local_2 = (flip_local_1 + CuArray(rand(1:L-1, N)) .- 1) .% L .+ 1
        #println("flip_local = ", flip_local_2)
        flip_global_1 = gen_flip_list(flip_local_1, L) # Global position of sites needed to flip 
	flip_global_2 = gen_flip_list(flip_local_2, L) # Global position of sites needed to flip
	#println("flip_global_1 = ", flip_global_1)
	#println("flip_global_2 = ", flip_global_2)
	s_sum = sample.state[flip_global_1[:,2]] .+ sample.state[flip_global_2[:,2]] .+ 1
	#println("s_sum = ",s_sum)
	#x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        
	p = abs.(Ratio_ψ(rbm, sample.state, flip_global_1, flip_global_2)).^2
        #println("p= ",p)
        r = CUDA.rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("Before sr= ",sr)
	sr = (transpose(s_sum) .== sr)
	#println("After sr= ",sr )
        s_ind = CuArray(findall(x->x==1,sr[:])) # Find the indics of configurations which accept the new configurations
        site_ind_accept_1 = flip_global_1[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
	site_ind_accept_2 = flip_global_2[s_ind,2]
	sample.state[site_ind_accept_1] .= -sample.state[site_ind_accept_1]
	sample.state[site_ind_accept_2] .= -sample.state[site_ind_accept_2]
    end
end

#=======================================#
#=======================================#

########### --- Computation: S Matrix --- ###########
########### --- Computation: S Matrix --- ###########

function compute_sr_gradient_G_gpu(S::CuArray{T, 2},sample::Abstract_Sampling) where T
    #S = regularize_G(S, sample)
    S += 1e-2 * I 
    Sinv = cuinv(S) 
    return Sinv
end

function regularize_G(S::CuArray{T, 2}, sample::Abstract_Sampling; λ₀=100, b=0.9, λ_min = 1e-4) where T
    p = sample.counter
    λ = x -> max(λ₀ * b^p, λ_min)
    Sreg = S + λ(p) * Diagonal(diag(S))
end

function cuinv(A::CuArray{T, 2}) where T
    if size(A, 1) != size(A, 2) throw(ArgumentError("Matrix not square.")) end

    B = CuArray(Matrix{T}(I(size(A,1))))
    A, ipiv = CUDA.CUSOLVER.getrf!(A) #LU factorization
    return CuMatrix{T}(CUDA.CUSOLVER.getrs!('N', A, ipiv, B)) # Solves a system of linear equations with an LU-factored square coefficient matrix, with multiple right-hand sides
end

#=======================================#
#=======================================#

########### --- Main Code: Optimization --- ###########
########### --- Main Code: Optimization --- ###########

export gs_main_run, Optimize_gs

function gs_main_run(Quantum_Model::T1, rbm::RBM{T2}, sample::Sampling, nm::Any, iterate_mode :: String, correlation_num :: Int) where{T1 <: Abstract_Quantum_System, T2 <: Number}
    if nm == "all"
        #l_nm = rbm._nv + rbm._nv * rbm._nh #updat all parameters
        l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters
    #elseif nm == 0
    #    l_nm = rbm._nv + rbm._nh #update a and b
    elseif typeof(nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
	    #l_nm =  length(nm)*rbm._nh
	    #l_nm = length(nm) + length(nm) * rbm._nh # update a and W
        l_nm = length(nm) + rbm._nh + length(nm) * rbm._nh # update a ,b and W
    else
	    error("The setting of Block is wrong! ====> $nm")
    end

    println("Iteration:  $iterate_mode")
    println("l_nm = ",l_nm)
    E_loc_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm)) 
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    ∂θ_initial = CuArray(zeros(T2,l_nm, sample.N_states))
    col_avg = CuArray(zeros(Quantum_Model.n_sites, Quantum_Model.n_sites))

    #col_single_avg = CuArray(zeros(Quantum_Model.n_sites))

    tot_time_S = 0.0
    tot_time_E_loc = 0.0
    tot_time_S = @elapsed for j = 1 : sample.n_sweeps
        Single_Sweep!(Quantum_Model, rbm, sample)
        #s.state = [1,-1,1,-1]
        #println("state= ",s.state)
        tot_time_E_loc_m = @elapsed E_loc, ∂θ_nm, E_loc∂θ  = measure_G(Quantum_Model, rbm, sample.state, ∂θ_initial, sample.N_states, nm)
        #push!(∂θ_list, ∂θ)
        E_loc_avg += sum(E_loc)
        S_kk .=0

	    tot_time_E_loc += tot_time_E_loc_m 
        #time_S = @elapsed    CUDA.@sync begin
	    #@btime CUDA.@sync begin
	    #@time begin
	    #CUDA.@sync begin
        #    @cuda  blocks = 8192 threads = 1024  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, sample.N_states);
        #end
        #end
        #elapsed btime
        #tot_time_S += time_S
        #println("time of building matrix S = ", time_S)
        #GC.gc()
        #CUDA.reclaim()
        #∂θ_Matrix .+= S_kk
	    ∂θ_Matrix .+= conj(∂θ_nm) * transpose(∂θ_nm)

        ∂θ_avg += sum(∂θ_nm,dims = 2)
        E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
        for site_j = 1 : Quantum_Model.n_sites
		    col_avg[:, site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
	    end
        #	col_single_avg += sum(sample.state, dims = 2)

    end
    println("Time of Sampling = ", tot_time_S)
    println("Time of E_loc = ", tot_time_E_loc)
    # ∂θ_Matrix = nothing
    # S_kk = nothing
    # ∂θ_initial = nothing
    # ∂θ_nm = nothing
    # GC.gc()
    # CUDA.reclaim()

    N_total = sample.n_sweeps * sample.N_states

    E_loc_avg /=  N_total
    ∂θ_avg ./= N_total
    E_loc∂θ_avg ./= N_total
    ∂θ_Matrix ./= N_total

    col_avg ./= N_total
    #col_single_avg ./= N_total
    #    correlation_avg = col_avg .- col_single_avg[correlation_num] .* col_single_avg
    #S = compute_Skl_GS(∂θ_list)
   # println(size(∂θ_avg))
    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
    GC.gc()
    CUDA.reclaim()
    # println("type of S =   ", typeof(S))
    time_inv = @elapsed Sinv = compute_sr_gradient_G_gpu(S,sample)
    S = nothing
    GC.gc()
    CUDA.reclaim()
    gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #     gNQS = (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    ∂θ_avg = nothing
    E_loc∂θ_avg = nothing
    
    GC.gc()
    CUDA.reclaim()

    println("Total time of sampling: ", tot_time_S)
    println("Total time of inverse: ", time_inv)
    
    return E_loc_avg, col_avg, gNQS[:]

end

function gs_main_run(Quantum_Model::T1, rbm::RBM{T2}, sample::exact_sampling, nm::Any, iterate_mode :: String, correlation_num :: Int) where{T1 <: Abstract_Quantum_System, T2 <: Number}
    if nm == "all"
        #l_nm = rbm._nv + rbm._nv * rbm._nh #updat all parameters
        l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters
    #elseif nm == 0
    #    l_nm = rbm._nv + rbm._nh #update a and b
    elseif typeof(nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
	    #l_nm =  length(nm)*rbm._nh
	    #l_nm = length(nm) + length(nm) * rbm._nh # update a and W
        l_nm = length(nm) + rbm._nh + length(nm) * rbm._nh # update a ,b and W
    else
	    error("The setting of Block is wrong! ====> $nm")
    end

    println("Iteration:  $iterate_mode")
    println("l_nm = ",l_nm)
    E_loc_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm)) 
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    ∂θ_initial = CuArray(zeros(T2,l_nm, sample.n_states))
    col_avg = CuArray(zeros(Quantum_Model.n_sites, Quantum_Model.n_sites))

    exact_ψ = ψ_G(rbm, sample.state, sample.n_states)
    P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)   
    E_loc, ∂θ_nm, E_loc∂θ  = measure_G(Quantum_Model, rbm, sample.state, ∂θ_initial, sample.n_states, nm)
    
    E_loc_avg += sum(E_loc .* P_ψ)
    ∂θ_Matrix .+= conj(P_ψ .* ∂θ_nm) * transpose(∂θ_nm)
    ∂θ_avg += sum(P_ψ .* ∂θ_nm, dims = 2)
    E_loc∂θ_avg += sum(P_ψ .* E_loc .* conj(∂θ_nm), dims = 2)
    for site_j = 1 : Quantum_Model.n_sites
        col_avg[:, site_j] = sum(sample.state[site_j,:]' .* sample.state, dims = 2)
    end
    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
    GC.gc()
    CUDA.reclaim()
    # println("type of S =   ", typeof(S))
    #Sinv = compute_sr_gradient_G_gpu(S,sample)
    S += 1e-2 * I
    gNQS = S \ (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    S = nothing
    GC.gc()
    CUDA.reclaim()
    #gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    ∂θ_avg = nothing
    E_loc∂θ_avg = nothing
    
    GC.gc()
    CUDA.reclaim()
    
    return E_loc_avg, col_avg, gNQS[:]

end

function gs_main_run(Quantum_Model::T1, rbm::RBM{T2}, sample::Sampling, nm::Any, iterate_mode :: Int, correlation_num :: Int) where{T1 <: Abstract_Quantum_System, T2 <: Number}
    if nm == "all"
        #l_nm = rbm._nv  + rbm._nv * rbm._nh #updat all parameters
        l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters
    #elseif nm == 0
    #    l_nm = rbm._nv + rbm._nh #update a and b
    elseif typeof(nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
	    #l_nm =  length(nm)*rbm._nh
	    #l_nm = length(nm) + length(nm) * rbm._nh # update a and W
        l_nm = length(nm) + rbm._nh + length(nm) * rbm._nh # update a, b and W
    else
        error("The setting of Block is wrong!")
    end


    println("Iteration: $iterate_mode")

    E_loc_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm)) 
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    ∂θ_initial = CuArray(zeros(T2,l_nm, sample.N_states))
    col_avg = CuArray(zeros(Quantum_Model.n_sites, Quantum_Model.n_sites))
    #col_single_avg = CuArray(Quantum_Model.n_sites)

    for j = 1 : sample.n_sweeps
        Single_Sweep!(Quantum_Model, rbm, sample)
        #s.state = [1,-1,1,-1]
        #println("state= ",s.state)
        E_loc, ∂θ_nm, E_loc∂θ  = measure_G(Quantum_Model, rbm, sample.state, ∂θ_initial, sample.N_states, nm)
        #push!(∂θ_list, ∂θ)
        E_loc_avg += sum(E_loc)

    #        S_kk .=0
    #        CUDA.@sync begin
    #            @cuda  blocks = 1024 threads = 1024  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, sample.N_states);
    #        end
            #GC.gc()
            #CUDA.reclaim()
    #        ∂θ_Matrix .+= S_kk
        for site_j = 1 : Quantum_Model.n_sites
        	col_avg[:,site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
	end

    #   col_single_avg += sum(sample.state, dims = 2)

        ∂θ_avg += sum(∂θ_nm,dims = 2)
        E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
    end
    # ∂θ_Matrix = nothing
    # S_kk = nothing
    # ∂θ_initial = nothing
    # ∂θ_nm = nothing
    # GC.gc()
    # CUDA.reclaim()

    N_total = sample.n_sweeps * sample.N_states

    E_loc_avg /=  N_total
    ∂θ_avg ./= N_total
    E_loc∂θ_avg ./= N_total
    col_avg ./= N_total
    #   col_single_avg ./= total
    #   correlation_avg = col_avg .- col_single_avg[correlation_num] .* col_single_avg
    #   ∂θ_Matrix ./= N_total
        #S = compute_Skl_GS(∂θ_list)
    # println(size(∂θ_avg))
    #    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg) 
    # println("type of S =   ", typeof(S))
    #    Sinv = compute_sr_gradient_G_gpu(S,sample)
    #    gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
     gNQS = (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    θ_avg = nothing
    E_loc∂θ_avg = nothing
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
    GC.gc()
    CUDA.reclaim()

    
    return E_loc_avg, col_avg,  gNQS[:]

end

function Optimize_gs(Quantum_Model::T1, Lx::Int, Ly::Int, Neural_Net::T2, sample::T_S, block_size_xy::Array{Int,1}, num_epochs::Int, n_loop::Int, N_Shift::Int, update_mode::String, LR_coff::Float64, final_γ :: Float64, segment_redu_γ::Int, iterate_mode, random_seed_num::Int, correlation_num :: Int) where {T1<:Abstract_Quantum_System, T2 <: Abstract_Learning_Machine, T_S <: Abstract_Sampling}
    #∂θ_up = CuArray(zeros(ComplexF64,nm,1))
    E_GS = 0.0+0.0im
    Block_ind = []
    block_size = block_size_xy[1] 
    if update_mode == "all"
        push!(Block_ind, "all")
        N_blocks = 1
        println("All parameters will be updated")
    elseif update_mode == "block"
	    	if Ly == 1 # 1D case
        		#push!(Block_ind, 0) # generate the index of Block, the first index is 0, which means a and b parameters are updated.
			N_blocks = ceil(Int,Quantum_Model.n_sites/block_size) # block_size means how many sites in every block 
			#        for j = 1 : N_blocks
		#            j < N_blocks ? push!(Block_ind,CuArray(collect((j-1)*block_size+1:j*block_size))) : push!(Block_ind,CuArray(collect((j-1)*block_size+1:Quantum_Model.n_sites)))
		#	  end
	        if mod(Quantum_Model.n_sites , 2) == 0
				N_blocks_tot = 2 * N_blocks - 1
			else
				N_blocks_tot = 2 * N_blocks - 2
			end

			for j = 1 : N_blocks_tot
				block_size_2 = Int(block_size/2)
				push!(Block_ind, CuArray(collect( (j-1)*block_size_2 + 1 : block_size_2 * (j+1)  )))
			end
		elseif Ly != 1 # 2D case
			if block_size_xy == [2, 1]
				Block_ind = generate_block_2D(Quantum_Model)
			else
				Block_ind = generate_block_2D(Lx, Ly, block_size_xy) 
			end
		end
        Block_ind = [Block_ind ; reverse(Block_ind)]
	pop!(Block_ind) #remove the first block from Block index
	deleteat!(Block_ind, Int((length(Block_ind) + 1)*0.5)) #remove the last block from Block index
#        println("size of Block = ", Int(size(Block_ind,1)/2))
        println("number of Block = ", Int(size(Block_ind,1)))
    else
        error("There isn't such option: $update_mode. The option of update mode are: all and block.")
    end
    Block_ind_0 = Block_ind
    Loop = 1
    Total_sites = Quantum_Model.n_sites
    if typeof(sample) == Sampling
        Thermalization!(Quantum_Model, Neural_Net, sample)
    end

    Total_shift = 0
    Total_sample = 1
    nn_α = Int(Neural_Net._nh/Neural_Net._nv)
    n_iterations = 0
    sample.counter = 500
    println("Counter = ", sample.counter)
    while  Loop <= n_loop
        println("Loop = ",Loop)
        #if (Loop - 1)  % segment_redu_γ == 0 && Loop != 1 && Neural_Net.α> final_γ + 1e-9
	    #Neural_Net.α = Neural_Net.α *LR_coff
        #    println("Learning rate = ", Neural_Net.α)
        #end
        #if  update_mode == "block" && N_Shift > 0
        if update_mode == "block"
            Block_ind = generate_new_Block(Block_ind_0, mod(Total_shift, block_size), Quantum_Model.n_sites)
        end
        println("Shift = ",Total_shift)
        println(Block_ind)
        # println(typeof(Block_ind[2]))
        for m = 1 : length(Block_ind)
            block_m = Block_ind[m]
            time_iteration = @elapsed    for i = 1:num_epochs # how many updates we have before move to the next Block
            n_iterations += 1
            println("Number of iterations = $n_iterations")
            if (n_iterations - 1)  % segment_redu_γ == 0 && n_iterations != 1 && Neural_Net.α> final_γ + 1e-9
                Neural_Net.α = Neural_Net.α *LR_coff
                println(">>>>Learning rate = ", Neural_Net.α)
            end
            sample.counter += 1
            #∂θ_up .=0
            #E_avg,  ∂θ  = run_G_device_1(s_g, nm, N)
            sum_NN = sum(abs.(Neural_Net.W[:,:])) + sum(abs.(Neural_Net.a[:]))
            println("sum_NN = ",sum_NN)
            E_avg, corr_avg, ∂θ  = gs_main_run(Quantum_Model, Neural_Net, sample, block_m, iterate_mode, correlation_num) 
            if abs(E_avg) >= 1e+8
                error("Energy = $E_avg   Wrong result !!!")
            end
            #println(size(∂θ))
            #E_avg,  ∂θ  = run_G_abW(s_g, nm, N) 
            if i == num_epochs
                println("Loop = ",Loop, "  nm = ",block_m)
                println(i,"  th  =====>>>  Energy = ",E_avg)
                Print_txt(Loop, E_avg, Total_sites, nn_α, block_size, Total_sample, random_seed_num)
                println("size of correlation = ",size(corr_avg))
                Print_txt_corr(Loop, corr_avg, Total_sites, nn_α, block_size, Total_sample, random_seed_num)
                #Print_txt(Loop,  n_sites, α, Total_sample, random_seed_num, time_iteration)
            end  
                update_parameters!(Neural_Net, ∂θ, block_m)
 
                #t>0 ? Adam_update!(adam, s_g.rbm_g, ∂θ[:,1], t, nm) : update_G!(s_g.rbm_g, ∂θ, nm)
            end
	    Print_txt(Loop, Total_sites, nn_α, block_size, Total_sample, random_seed_num, time_iteration)
        end
	#Print_txt(Loop,  n_sites, α, Total_sample, random_seed_num, time_iteration)
        println("=======================================")

        if update_mode == "block"
            Total_shift = mod(Total_shift + N_Shift, block_size)
        end

        
        # N_Shift = mod(Int(Loop/2),2)
        
        #t += 1
        Loop += 1

    end
    #return res[end]
end

function generate_new_Block(Block, N_shift :: Int, nv ::Int)
    Block_new = []
    for n = 1 : length(Block)
        #if (n == 1 || n == length(Block) )
        #    push!(Block_new,Block[n])
        #else
            block_shift_n = Block[n] .+ N_shift
            block_shift_n = mod.(block_shift_n .-1, nv) .+1
            push!(Block_new, block_shift_n)
        #end
    end
    return Block_new
end

# --------------------------------------------
# --------------------------------------------

# --------------------------------------------
# --------------------------------------------

function update_parameters!(rbm :: RBM{T}, ∂θ::CuArray{T,1}, nm::String) where T # update all parameters
    #print("n_visible= ",∂θ[1])
    rbm.a .-= rbm.α * ∂θ[1:rbm._nv]
    rbm.b .-= rbm.α * ∂θ[rbm._nv+1:rbm._nv+rbm._nh] 
    rbm.W .-= rbm.α * reshape(∂θ[rbm._nv + rbm._nh + 1 : end], (rbm._nh, rbm._nv))
    #rbm.W .-= rbm.α * reshape(∂θ[rbm._nv + 1 : end], (rbm._nh, rbm._nv))
    #return 
end

function update_parameters!(rbm :: RBM{T}, ∂θ::CuArray{T,1}, nm::Int) where T # update a and b
    #print("n_visible= ",∂θ[1])
    rbm.a .-= rbm.α * ∂θ[1:rbm._nv]
    rbm.b .-= rbm.α * ∂θ[rbm._nv+1:rbm._nv+rbm._nh] 

end

function update_parameters!(rbm_g :: RBM{T}, ∂θ::CuArray{T,1}, nm::CuArray{Int,1}) where T # update a, b and W
    rbm_g.a[nm] .-= rbm_g.α * ∂θ[1:length(nm)] 
    rbm.b .-= rbm.α * ∂θ[length(nm) + 1 : length(nm) + rbm._nh] 
    rbm_g.W[:, nm] .-= rbm_g.α * reshape(∂θ[length(nm) + rbm._nh + 1 : end], (rbm_g._nh, length(nm)))
end

# --------------------------------------------
# --------------------------------------------


function Print_txt(Loop, Energy, n_sites, α, block_size, Total_sample, random_seed_num)
    io = open("Ground_Energy_$(n_sites)_sites_$(α)_$(block_size)_$(Total_sample)_$(random_seed_num).txt", "a+");
    println(io, Loop,"   ", Energy)
    close(io)
end

function Print_txt_corr(Loop, correlation, n_sites, α, block_size, Total_sample, random_seed_num)
    io2 = open("Correlation_$(n_sites)_sites_$(α)_$(block_size)_$(Total_sample)_$(random_seed_num).txt", "a+");
    println(io2, Loop,"   ", correlation)
    close(io2)
end


function Print_txt(Loop, n_sites, α, block_size, Total_sample, random_seed_num, time_iteration :: Float64)
    io = open("Ground_Energy_$(n_sites)_sites_$(α)_$(block_size)_$(Total_sample)_$(random_seed_num).txt", "a+");
     println(io , "t: ", time_iteration)
     close(io)
end

#=======================================#
#=======================================#

########### --- Main Code: Initialization --- ###########
########### --- Main Code: Initialization --- ###########

export Neural_Net_Initialization
function Neural_Net_Initialization(::Type{T}, quan_sys_setting::Tuple, neural_net_setting::Tuple, sample_setting::Tuple, update_setting::Tuple) where{T<:Number}
    
    # basic setting
    quan_sys_name = quan_sys_setting[1]
    Lattice = quan_sys_setting[2]
    Lx = Lattice[1]
    Ly = Lattice[2]
    n_sites = Lx*Ly

    # set quantum model
    if quan_sys_name == "Ising"

        J = quan_sys_setting[3]
        hx = quan_sys_setting[4]
        hz = quan_sys_setting[5]
        boundary_condition = quan_sys_setting[6]
	    correlation_num = quan_sys_setting[7]
        Quantum_Model = Ising_Model_Initialization(J, hx, hz, Lx, Ly, boundary_condition)
        #ini_states = CuArray(rand(-1:2:1, n_sites, n_states))
        #sample = Sampling(n_thermals, n_sweeps, N, 0, ini_states)
    elseif quan_sys_name == "Heisenberg"
        J = quan_sys_setting[3]
        boundary_condition = quan_sys_setting[4]
        Quantum_Model = Heisenberg_Model_Initialization(J, Lx, Ly, boundary_condition)
        #ini_states = CuArray(rand(-1:2:1, n_sites, N))
        #sample = Sampling(n_thermals, n_sweeps, N, 0, ini_states)
    else
        error("There is no such a model: $quan_sys_name. The options of quantum system are: Ising and Heisenberg.")
    end

   # load_NN_p = NN_parameters[1]
   # if load_NN_p == 0
#	    println("No need to load initial parameters:  ",Quantum_Model.a[1] )
#   elseif load_NN_p == 1
#	    Quantum_Model.a = CuArray(NN_parameters[2])
#	    Quantum_Model.W = CuArray(NN_parameters[3])
#	    pritnln("The initial parameters are loaded:  a[1] = $(Quantum_Model.a[1]),    W[1] = $(Quantum_Model.W[1])")
#   else 
#	    error("0 means no parameters loaded, 1 means parameters are needed to load")
#    end
    

    sample_name = sample_setting[1]
    n_thermals = sample_setting[2]
    n_sweeps = sample_setting[3]
    n_states = sample_setting[4]
    if sample_name == "metropolis"
        
        init_states = CuArray(rand(-1:2:1, n_sites, n_states))
        sample = Sampling(n_thermals, n_sweeps, n_states, 0, init_states)
        n_total = n_sweeps * n_states
    elseif sample_name == "exact"
        if n_sites > 16
            error("The number of sites is too large, we can not do exact sampling.")
        end
        n_exact_states = 2^(Quantum_Model.n_sites) # number of states for exact sampling
        exact_states = zeros(Int, Quantum_Model.n_sites, n_exact_states)
        n_sweeps = n_exact_states
        for j = 0 : n_exact_states -1
            string_state = string(j, base = 2, pad = Quantum_Model.n_sites)
            for k = 1 : Quantum_Model.n_sites
                exact_states[k,j+1] = parse(Int,string_state[k])
            end
        end
        exact_states .= -2exact_states .+ 1
        n_states = n_exact_states
        sample = exact_sampling(0, n_states, CuArray(exact_states))
        n_total = n_states
    else
        error("There is no such a sampling method: $sample_name. The options of sampling method are: metropolis, exact.")
    end


    println("bond of y direction = ", Quantum_Model.site_bond_y)
    # set neural network
    neural_net_name = neural_net_setting[1]
   
    if neural_net_name == "RBM"
        nv = n_sites
        nn_α = neural_net_setting[2]
        nh = nn_α * nv
        α = neural_net_setting[3]
	random_seed_num = neural_net_setting[4]
        Neural_Net = RBM_Initialization(T, nv, nh, α, random_seed_num)
	println("====>>>>>>", typeof(Neural_Net.a))
    else
        error("There is no such a neural network: $neural_net_name. The options of quantum system are: RBM.")
    end

    # load_NN_p = NN_parameters[1]
    # if load_NN_p == 0
	#     println("No need to load initial parameters:  ",Neural_Net.a[1] )
    # elseif load_NN_p == 1
	#     Neural_Net.a = CuArray(NN_parameters[2])
	#     Neural_Net.W = CuArray(NN_parameters[3])
	#     println("The initial parameters are loaded:  a[1] = $(Neural_Net.a[1]),    W[1] = $(Neural_Net.W[1])")
    # else
	#     error("0 means no parameters loaded, 1 means parameters are needed to load")
    # end





    update_mode = update_setting[1]
    block_size_xy = update_setting[2]
    block_size = block_size_xy[1] # block size on x direction
    num_epochs = update_setting[3]
    n_loop = update_setting[4]
    n_shift_site = update_setting[5]
    redu_coff_α = update_setting[6]
    final_γ = update_setting[7]
    segment_redu_γ = update_setting[8]
    iteration_mode = update_setting[9]

#    if Ly == 1 # 1D case
#	    Block_ind_xy = []
#    else
#	    Block_ind_xy = generate_block_2D(Lx, Ly, block_size)
#    end

    if update_mode == "all" && (block_size != 0 || n_shift_site != 0 )
        error("When the mode of updating is all, the size for each block should be 0, instead of $block_size. And n_shift_site should be 0.")
    elseif update_mode == "block" && (block_size > n_sites )
        error("Block size exceeds the total number of lattices.")
    end

    if iteration_mode != "SR" && iteration_mode != 0
	    error("There is no such iteration mode. /the option are: SR and 0 wihch means normal method without SR.")
    end

    # record settings
    io = open("Ground_Energy_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "w");
    io = open("Ground_Energy_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "a+");
    println(io, "Date:  ",(today()),"  ", Dates.Time(Dates.now()))
    println(io, "The ground energy of $quan_sys_name model, and the total site is $n_sites, where Lx = $Lx, Ly = $Ly. ")
    println(io, "The boundary condition is: $boundary_condition") 
    if quan_sys_name == "Ising"
	    println(io,"J = $J,   hx = $hx")
	    println(io, "hz = $hz \n")
    end
    if neural_net_name == "RBM"
        println(io, "for RBM, nv = $n_sites, nh = $nh. Learning rate γ = $α, The cofficient of reducing learning rate redu_coff_γ = $redu_coff_α, final learning rate = $final_γ, segment_redu_γ = $segment_redu_γ")
    end
    
    println(io, "The mode of optimization is $update_mode, and the block size_x = $block_size, the block size on y direction = $(block_size_xy[2]) the shift amount = $n_shift_site.")
    
    println(io, "For the sampling, the number state updated in one step is: $n_states, the number of thermailization is: $n_thermals, the number of sweeps is $n_sweeps.")
    println(io, "Thus, the total state is $(n_states * n_sweeps). \n")
    println(io,"The random seed number is:  $random_seed_num")
    CUDA.@allowscalar println(io, "a[1] = ", Neural_Net.a[1], "  W[1] = ", Neural_Net.W[1])

    println(io, "Initial Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)
    
    io2 = open("Correlation_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "w");
    io2 = open("Correlation_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "a+");
    println(io2, "Correlation Number =  $correlation_num  ")
    close(io2)

    Optimize_gs(Quantum_Model, Lx, Ly, Neural_Net, sample, block_size_xy, num_epochs, n_loop, n_shift_site, update_mode, redu_coff_α, final_γ, segment_redu_γ,  iteration_mode, random_seed_num, correlation_num)
	
    if neural_net_name == "RBM"
	    save("RBM_GS_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).jld2","RBM_GS_a",Array(Neural_Net.a), "RBM_GS_b", Array(Neural_Net.b), "RBM_GS_W",Array(Neural_Net.W))
    end
    io = open("Ground_Energy_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "a+");
    println(io, "Finish Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)

    io = open("Ground_Energy_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(n_states * n_sweeps)_$(random_seed_num).txt", "a+");
    println(io, "The final learning rate is:  ", Neural_Net.α)
    close(io)

   # println("a = ", typeof(Neural_Net.a))
    if quan_sys_name == "Ising" && neural_net_name == "RBM"        
       save("./$quan_sys_name, L = $n_sites,  Lx_$Lx,  Ly = $Ly,  J = $J, hx = $hx.jld2","a0",Array(Neural_Net.a),"b0",Array(Neural_Net.b),"W0",Array(Neural_Net.W))
    end
    

end
