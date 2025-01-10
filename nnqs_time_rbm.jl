# This code if for p-tVMC and tVMC with RBM:

using Base: Float64
using CUDA
using LinearAlgebra
using Statistics
using Distributions 
using Random
using Dates
using JLD2



export  Neural_Net_Initialization_parallel_Time


abstract type Abstract_Quantum_System end # Define type of quantum systems: Ising model, Heisenberg models

abstract type Abstract_Learning_Machine{T <: Number} end # Define neural network models: Restricted Boltzmann Machine, Convetional Neural Network ...

abstract type Abstract_Sampling end # Define sampling methods: Monte Carlo, Metropolis Hsating ...

abstract type abstract_evolution end### ------- type of evolution -------
    


###########  ---Neural Network Models --- ###########
###########  ---Neural Network Models --- ###########

export RBM, RBM_Initialization, RBM_Time, measure_time_SR_Trotter, Ratio_ψϕ, Ratio_ϕψ	
mutable struct RBM{T} <: Abstract_Learning_Machine{T} 
    a::CuArray{T, 1}
    b::CuArray{T, 1}
    W::CuArray{T, 2}
    _nv::Int
    _nh::Int
    γ::Float64
end

mutable struct RBM_Time{T} <: Abstract_Learning_Machine{T} # Parameters of time evolution
    a::CuArray{T, 1}
    b::CuArray{T, 1}
    W::CuArray{T, 2}
end


mutable struct RBM_n{T} <: Abstract_Learning_Machine{T} # Parameters of time evolution
    a::CuArray{T, 1}
    b::CuArray{T, 1}
    W::CuArray{T, 2}
end


function RBM_Initialization(::Type{Float64}, nv::Int, nh::Int, α::Float64, random_seed_num::Int; sigma = 0.01)   # Initializing data of RBM if float number
    Random.seed!(random_seed_num)
    d = Normal(0, sigma)
    a = rand( d, nv)
    b = rand( d, nh)
    W = rand( d, nh, nv)
    
    return RBM{Float64}(a, b, W, nv, nh, α)
end

function RBM_Initialization(::Type{ComplexF64}, nv::Int, nh::Int, α::Float64, random_seed_num::Int; sigma = 0.01)   # Initializing data of RBM if complex number
    Random.seed!(random_seed_num)
    d = Normal(0, sigma)
    a = rand(d, nv)  .+ im * rand(d, nv)
    b = rand(d, nh)  .+ im * rand(d, nh)
    W = rand(d, nh, nv) .+ im * rand(d, nh, nv)

    return RBM{ComplexF64}(a, b, W, nv, nh, α)
    
end

function RBM_Time_Initialization(nv::Int, nh::Int, random_seed_num::Int; sigma = 1.0)
	Random.seed!(random_seed_num)
	d = Normal(0, sigma)
	a = rand( d, nv) .+ im * rand(d, nv)
    b = rand(d, nh)  .+ im * rand(d, nh)
	W = rand( d, nh, nv) .+ im * rand(d, nh, nv)
	return RBM_Time{ComplexF64}(a, b, W)
end

#=======================#
#=======================#

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
    return CuAttay(Site_Bond_x), CuArray(Site_Bond_y)
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

#=======================#
#=======================#

########### ---  Trotter Block  --- ###########
########### ---  Trotter Block  --- ###########

export Un, Un_m
mutable struct Un
    U::CuArray{ComplexF64, 2}
    U_dag::CuArray{ComplexF64, 2}
    ind_U_col::CuArray{Int64, 1}
    ind_state::CuArray{Int64,2}
end
function Un(ising::Ising_Model, dt:: Float64, n::Int)  
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
    return Un(CuArray(U), CuArray(U_dag), CuArray(ind_U_col))
end

function Un_m(ising::Ising_Model, dt:: Float64, n::Int, N_H_bond::Int)  # for 1D case 
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

########### --- Local Energy: E_loc --- ###########
########### --- Local Energy: E_loc --- ###########

export Ratio_ψ, gen_flip_list,ψ_G

function ψ_G(rbm_g::RBM, x::CuArray{Int64}, N::Int) # calculate wave function ψ using GPU
    aT = reshape(rbm_g.a,1,length(rbm_g.a))
    #b = rbm_g.b
    #b = repeat(rbm_g.b,1,N)
    #W = rbm_g.W
    #println("size of C",size(W * x))
    C = rbm_g.W * x .+ rbm_g.b
    
    return CUDA.reduce(*, [exp.(aT * x);  (exp.(C).+exp.(-C))],dims =1)
end


function ϕ_G(rbm_t::RBM_Time, x::CuArray{Int64}, N::Int) # calculate wave function ψ using GPU
    aT = reshape(rbm_t.a,1,length(rbm_t.a))
    #b = rbm_g.b
    #b = repeat(rbm_g.b,1,N)
    #W = rbm_g.W
    #println("size of C",size(W * x))
    C = rbm_t.W * x .+ rbm_t.b
    
    return CUDA.reduce(*, [exp.(aT * x);  (exp.(C).+exp.(-C))],dims =1)
end



function Ratio_ψ(rbm_g::RBM, x::CuArray{Int64, 2}, flip_list::CuArray{Int64,2}) # calculate wave function ψ on GPU for ground state
    Δa = -2*rbm_g.a[flip_list[:,1]] .* x[flip_list[:,2]]
    C = rbm_g.W * x .+ rbm_g.b
    C_new  = C .- 2*rbm_g.W[:,flip_list[:,1]] .* transpose(x[flip_list[:,2]])
    return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
    #return reduce(*, [transpose(exp.(Δa));  cosh.(C_new)./cosh.(C)],dims =1)
end

function Ratio_ψ(rbm_g::RBM, x::CuArray{Int64, 2}, flip_list_1::CuArray{Int64,2}, flip_list_2::CuArray{Int64,2}) # calculate wave function ψ on GPU
	Δa = -2*rbm_g.a[flip_list_1[:,1]] .* x[flip_list_1[:,2]] .- 2*rbm_g.a[flip_list_2[:,1]] .* x[flip_list_2[:,2]]
	C = rbm_g.W * x .+ rbm_g.b
	C_new  = C .- 2*rbm_g.W[:,flip_list_1[:,1]] .* transpose(x[flip_list_1[:,2]]) .- 2*rbm_g.W[:,flip_list_2[:,1]] .* transpose(x[flip_list_2[:,2]])
	return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
    #return reduce(*, [transpose(exp.(Δa));  cosh.(C_new)./cosh.(C)],dims =1)
end

function Ratio_ϕ(rbm_t::RBM_Time, x::CuArray{Int64, 2}, flip_list::CuArray{Int64,2}) # calculate wave function ψ on GPU for ground state
    Δa = -2*rbm_t.a[flip_list[:,1]] .* x[flip_list[:,2]]
    C = rbm_t.W * x .+ rbm_t.b
    C_new  = C .- 2*rbm_t.W[:,flip_list[:,1]] .* transpose(x[flip_list[:,2]])
    return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
    #return reduce(*, [transpose(exp.(Δa));  cosh.(C_new)./cosh.(C)],dims =1)
end

function Ratio_ϕ(rbm_t::RBM_Time, x::CuArray{Int64, 2}, flip_list_1::CuArray{Int64,2}, flip_list_2::CuArray{Int64,2}) # calculate wave function ψ on GPU
	Δa = -2*rbm_t.a[flip_list_1[:,1]] .* x[flip_list_1[:,2]] .- 2*rbm_t.a[flip_list_2[:,1]] .* x[flip_list_2[:,2]]
	C = rbm_t.W * x .+ rbm_t.b
	C_new  = C .- 2*rbm_t.W[:,flip_list_1[:,1]] .* transpose(x[flip_list_1[:,2]]) .- 2*rbm_t.W[:,flip_list_2[:,1]] .* transpose(x[flip_list_2[:,2]])
	return CUDA.reduce(*, [transpose(exp.(Δa));  (exp.(C_new) .+ exp.(-C_new))./(exp.(C) .+ exp.(-C))],dims =1)
    #return reduce(*, [transpose(exp.(Δa));  cosh.(C_new)./cosh.(C)],dims =1)

end

function Ratio_ϕ(rbm_0::RBM_Time, rbm_n ::RBM_n, x::CuArray{Int64, 2} )
	Δa = transpose(rbm_n.a .- rbm_0.a) * x
	C_0 = rbm_0.W * x .+ rbm_0.b
	C_n = rbm_n.W * x .+ rbm_n.b
	return CUDA.reduce(*, [exp.(Δa);  (exp.(C_n) .+ exp.(-C_n))./(exp.(C_0) .+ exp.(-C_0))],dims =1)
end




function Ratio_ϕψ( rbm_t::RBM_Time, rbm_g::RBM, x_ϕ::CuArray{Int64, 2}, x_ψ::CuArray{Int64,2}) # calculate wave function ψ on GPU for time evolution
    A_ϕ = transpose(rbm_t.a) * x_ϕ
    A_ψ = transpose(rbm_g.a) * x_ψ
    Δa = A_ϕ .- A_ψ
    #println(size(Δa))
    C_ϕ = rbm_t.W * x_ϕ .+ rbm_t.b
    C_ψ = rbm_g.W * x_ψ .+ rbm_g.b
    #println(size(C_ϕ))
    return CUDA.reduce(*, [exp.(Δa);  (exp.(C_ϕ) .+ exp.(-C_ϕ))./(exp.(C_ψ) .+ exp.(-C_ψ))], dims =1)
    #return reduce(*, [exp.(Δa);  cosh.(C_ϕ) ./cosh.(C_ψ) ], dims =1)
end

function Ratio_ψϕ(rbm_g::RBM, rbm_t::RBM_Time, x_ψ::CuArray{Int64,2}, x_ϕ::CuArray{Int64, 2}) # calculate wave function ψ on GPU time evolution
    A_ψ = transpose(rbm_g.a) * x_ψ
    A_ϕ = transpose(rbm_t.a) * x_ϕ
    Δa = A_ψ .- A_ϕ
    C_ψ = rbm_g.W * x_ψ .+ rbm_g.b
    C_ϕ = rbm_t.W * x_ϕ .+ rbm_t.b
    return CUDA.reduce(*, [exp.(Δa);  (exp.(C_ψ) .+ exp.(-C_ψ))./(exp.(C_ϕ) .+ exp.(-C_ϕ))], dims =1)
    #return reduce(*, [exp.(Δa);  cosh.(C_ψ) ./cosh.(C_ϕ) ], dims =1)
end



function Ratio_ϕψ( rbm_t::RBM_n, rbm_g::RBM, x_ϕ::CuArray{Int64, 2}, x_ψ::CuArray{Int64,2}) # calculate wave function ψ on GPU for time evolution
    A_ϕ = transpose(rbm_t.a) * x_ϕ
    A_ψ = transpose(rbm_g.a) * x_ψ
    Δa = A_ϕ .- A_ψ
    #println(size(Δa))
    C_ϕ = rbm_t.W * x_ϕ .+ rbm_t.b
    C_ψ = rbm_g.W * x_ψ .+ rbm_g.b
    #println(size(C_ϕ))
    return CUDA.reduce(*, [exp.(Δa);  (exp.(C_ϕ) .+ exp.(-C_ϕ))./(exp.(C_ψ) .+ exp.(-C_ψ))], dims =1)
    #return reduce(*, [exp.(Δa);  cosh.(C_ϕ) ./cosh.(C_ψ) ], dims =1)
end

function Ratio_ψϕ(rbm_g::RBM, rbm_t::RBM_n, x_ψ::CuArray{Int64,2}, x_ϕ::CuArray{Int64, 2}) # calculate wave function ψ on GPU time evolution
    A_ψ = transpose(rbm_g.a) * x_ψ
    A_ϕ = transpose(rbm_t.a) * x_ϕ
    Δa = A_ψ .- A_ϕ
    C_ψ = rbm_g.W * x_ψ .+ rbm_g.b
    C_ϕ = rbm_t.W * x_ϕ .+ rbm_t.b
    return CUDA.reduce(*, [exp.(Δa);  (exp.(C_ψ) .+ exp.(-C_ψ))./(exp.(C_ϕ) .+ exp.(-C_ϕ))],dims =1)
    #return reduce(*, [exp.(Δa);  cosh.(C_ψ) ./cosh.(C_ϕ) ], dims =1)
end



function gen_flip_list(flip_local::CuArray{Int,1}, n_sites::Int) # Generate a flip index
    N = length(flip_local)
    shift_num = CuArray(collect(0: n_sites : (N-1) * n_sites))
    flip_global = flip_local .+ shift_num
    flip_list = [flip_local  flip_global]
    return flip_list
end


function E_local_ψϕ_Trotter(ising::Ising_Model ,rbm::RBM, rbm_t::RBM_Time, Uk::Un, initial_state::CuArray{Int}, bond_ind::CuArray{Int,1})
    #n = size(initial_state,1)
    #bond_ind = [ising.site_bond_x[k,1], ising.site_bond_x[k,2]]
    #println("initial state = ",initial_state)
    E_ψϕ = CuArray(zeros(ComplexF64,1,size(initial_state,2)))
    local_state = initial_state[bond_ind,:]
    #println("initial size = ", size(local_state))
    #println("initial type = ", typeof(local_state))
    #println("local state = ",local_state)
    U_col = sum(Int, -0.5*(local_state.-1).*Uk.ind_U_col,dims=1).+1 # Convert local state to the number
    #println("-------------0")
    #println("U_col = ",U_col)
    #ϕₓ = ϕ_G(rbm_t, initial_state, N)
    #println("-------------1")
    H_size = 2^length(bond_ind) 
    for j = 1:H_size
	#initial_state[:,1] .= CuArray([-1,-1,1,1,1,1,1,1])	
        X_new = copy(initial_state)
        #println("-------------3")
        X_new[bond_ind,:] .= Uk.ind_state[:,j]
        #println("-------------4")
        #E_ψϕ .+= ψ_G(rbm, X_new, N)./ϕₓ .* Uk.U[j,U_col]
	
        E_ψϕ .+= Ratio_ψϕ(rbm, rbm_t, X_new, initial_state) .* Uk.U[U_col, j]
	#println("rbm.a[1] = ",rbm.a[1])
	#println("rbm_t.a[1] = ",rbm_t.a[1])
	#println("$j  state_new[:,1] =",X_new[:,1])
	#println("$j  state[:,1] =", initial_state[:,1])
	#aaa = Ratio_ψϕ(rbm, rbm_t, X_new, initial_state)
	#println("$j ==> Ratio_ψϕ[1] = ",aaa[1])
	#sleep(2)
        #println("-------------5")
    end
    #println("===>  E_ψϕ[1] = ",E_ψϕ[1] )
    return E_ψϕ
end

function E_local_ϕψ_Trotter(ising::Ising_Model ,rbm::RBM, rbm_t::RBM_Time, Uk::Un, initial_state::CuArray{Int}, bond_ind::CuArray{Int,1})
    #n = size(initial_state,1)
    #bond_ind = [ising.site_bond_x[k,1], ising.site_bond_x[k,2]]
    #println("initial state = ",initial_state)
    E_ϕψ = CuArray(zeros(ComplexF64,1,size(initial_state,2)))
    local_state = initial_state[bond_ind,:]
    #U_dag = Uk.U #transpose(conj(Uk.U))
    U_col = sum(Int, -0.5*(local_state.-1).*Uk.ind_U_col,dims=1).+1 #Convert local state to the number
    #ψₓ = ψ_G(rbm, initial_state, N)
    H_size = 2^length(bond_ind)
    for j = 1:H_size
        X_new = copy(initial_state)
        X_new[bond_ind,:] .= Uk.ind_state[:,j]
        #E_ϕψ .+= ϕ_G(rbm_t, X_new, N)./ψₓ .* U_dag[j,U_col]
        E_ϕψ .+= Ratio_ϕψ(rbm_t, rbm, X_new, initial_state) .* Uk.U_dag[U_col, j]
        #E_ϕψ += sum(E_state)
    end
    return E_ϕψ
end

function E_local_ψϕ_Trotter(ising::Ising_Model ,rbm::RBM, rbm_t::RBM_n, Uk::Un, initial_state::CuArray{Int}, bond_ind::CuArray{Int,1})
    #n = size(initial_state,1)
    #bond_ind = [ising.site_bond_x[k,1], ising.site_bond_x[k,2]]
    #println("initial state = ",initial_state)
    E_ψϕ = CuArray(zeros(ComplexF64,1,size(initial_state,2)))
    local_state = initial_state[bond_ind,:]
    #println("initial size = ", size(local_state))
    #println("initial type = ", typeof(local_state))
    #println("local state = ",local_state)
    U_col = sum(Int, -0.5*(local_state.-1).*Uk.ind_U_col,dims=1).+1 # Convert local state to the number
    #println("-------------0")
    #println("U_col = ",U_col)
    #ϕₓ = ϕ_G(rbm_t, initial_state, N)
    #println("-------------1")
    H_size = 2^length(bond_ind) 
    for j = 1:H_size
	#initial_state[:,1] .= CuArray([-1,-1,1,1,1,1,1,1])	
        X_new = copy(initial_state)
        #println("-------------3")
        X_new[bond_ind,:] .= Uk.ind_state[:,j]
        #println("-------------4")
        #E_ψϕ .+= ψ_G(rbm, X_new, N)./ϕₓ .* Uk.U[j,U_col]
	
        E_ψϕ .+= Ratio_ψϕ(rbm, rbm_t, X_new, initial_state) .* Uk.U[U_col, j]
	#println("rbm.a[1] = ",rbm.a[1])
	#println("rbm_t.a[1] = ",rbm_t.a[1])
	#println("$j  state_new[:,1] =",X_new[:,1])
	#println("$j  state[:,1] =", initial_state[:,1])
	#aaa = Ratio_ψϕ(rbm, rbm_t, X_new, initial_state)
	#println("$j ==> Ratio_ψϕ[1] = ",aaa[1])
	#sleep(2)
        #println("-------------5")
    end
    #println("===>  E_ψϕ[1] = ",E_ψϕ[1] )
    return E_ψϕ
end

function E_local_ϕψ_Trotter(ising::Ising_Model ,rbm::RBM, rbm_t::RBM_n, Uk::Un, initial_state::CuArray{Int}, bond_ind::CuArray{Int,1})
    #n = size(initial_state,1)
    #bond_ind = [ising.site_bond_x[k,1], ising.site_bond_x[k,2]]
    #println("initial state = ",initial_state)
    E_ϕψ = CuArray(zeros(ComplexF64,1,size(initial_state,2)))
    local_state = initial_state[bond_ind,:]
    #U_dag = Uk.U #transpose(conj(Uk.U))
    U_col = sum(Int, -0.5*(local_state.-1).*Uk.ind_U_col,dims=1).+1 #Convert local state to the number
    #ψₓ = ψ_G(rbm, initial_state, N)
    H_size = 2^length(bond_ind)
    for j = 1:H_size
        X_new = copy(initial_state)
        X_new[bond_ind,:] .= Uk.ind_state[:,j]
        #E_ϕψ .+= ϕ_G(rbm_t, X_new, N)./ψₓ .* U_dag[j,U_col]
        E_ϕψ .+= Ratio_ϕψ(rbm_t, rbm, X_new, initial_state) .* Uk.U_dag[U_col, j]
        #E_ϕψ += sum(E_state)
    end
    return E_ϕψ
end


# function measure_time_SR_Trotter(rbm::RBM_G, rbm_t::RBM_Time, Uk::Un, state, state_t, k, ∂θ, nm, N)
#     E_ψϕ = E_local_ψϕ_Trottter(rbm, rbm_t, Uk, state_t, k, N)
#     E_ϕψ = E_local_ϕψ_Trotter(rbm, rbm_t, Uk, state, k, N)
#     #E_ψϕ = E_local_ψϕ_App(rbm, rbm_t, Uk, state_t, k)
#     #E_ϕψ = E_local_ϕψ_App(rbm, rbm_t, Uk, state, k)
#     E_loc_t = -E_ψϕ .* E_ϕψ 
#     if nm == "all"
#         ∂θ_t = Compute_derivatives_G(rbm_t, state_t, N)
#     else 
#         ∂θ_t = Compute_derivatives_G(rbm_t, state_t, ∂θ, nm, N)
#     end
    
#     return E_loc_t, ∂θ_t, E_loc_t .* conj(∂θ_t)
# end

function measure_time_SR_Trotter(ising :: Ising_Model, rbm::RBM, rbm_t::RBM_Time, Uk::Un, state::CuArray{T}, state_t::CuArray{T}, bond_ind::CuArray{T,1}, block_nm::CuArray{T,1}) where{T<:Int}
    E_ψϕ = E_local_ψϕ_Trotter(ising, rbm, rbm_t, Uk, state_t, bond_ind)
    E_ϕψ = E_local_ϕψ_Trotter(ising, rbm, rbm_t, Uk, state, bond_ind)
    #E_ψϕ = E_local_ψϕ_App(rbm, rbm_t, Uk, state_t, k)
    #E_ϕψ = E_local_ϕψ_App(rbm, rbm_t, Uk, state, k)
    #println("E_ψϕ[1] = $(E_ψϕ[1])")
    #println("E_ϕψ[1] = $(E_ϕψ[1])")
    E_loc_t = -E_ψϕ .* E_ϕψ 
    ∂θ_t = Compute_derivatives_G(rbm_t.b[:], rbm_t.W[:,:], state_t, size(state_t,2), block_nm)
    #bond_ind = [k, rbm.nv_bond[k]]
    #∂θ_t = Compute_derivatives_G(rbm_t, state_t, N, bond_ind, bond_b)
    E_loc_t1 =  (E_ψϕ)
    E_loc_t2 =  (E_ϕψ)
    return E_loc_t, ∂θ_t, E_loc_t .* conj(∂θ_t)
    #return E_loc_t1, E_loc_t2
end


function measure_time_SR_Trotter_IS(ising :: Ising_Model, rbm::RBM, rbm_t::RBM_Time, Uk::Un, state::CuArray{T}, state_t::CuArray{T}, bond_ind::CuArray{T,1}, block_nm::CuArray{T,1}) where{T<:Int}
    E_ψϕ = E_local_ψϕ_Trotter(ising, rbm, rbm_t, Uk, state_t, bond_ind)
    E_ϕψ = E_local_ϕψ_Trotter(ising, rbm, rbm_t, Uk, state, bond_ind)
    #E_ψϕ = E_local_ψϕ_App(rbm, rbm_t, Uk, state_t, k)
    #E_ϕψ = E_local_ϕψ_App(rbm, rbm_t, Uk, state, k)
    #println("E_ψϕ[1] = $(E_ψϕ[1])")
    #println("E_ϕψ[1] = $(E_ϕψ[1])")
    #E_loc_t = -E_ψϕ .* E_ϕψ 
    ∂θ_t = Compute_derivatives_G(rbm_t.b[:], rbm_t.W[:,:], state_t, size(state_t,2), block_nm)
    #bond_ind = [k, rbm.nv_bond[k]]
    #∂θ_t = Compute_derivatives_G(rbm_t, state_t, N, bond_ind, bond_b)
    E_loc_t1 =  (E_ψϕ)
    E_loc_t2 =  (E_ϕψ)
    return E_loc_t, ∂θ_t, E_loc_t .* conj(∂θ_t)
    #return E_loc_t1, E_loc_t2
end




function measure_time_SR_Trotter_New(ising :: Ising_Model, rbm::RBM, rbm_t::RBM_Time, Uk::Un, state::CuArray{T}, state_t::CuArray{T}, bond_ind::CuArray{T,1}, block_nm::CuArray{T,1}) where{T<:Int}
    E_ψϕ = E_local_ψϕ_Trotter(ising, rbm, rbm_t, Uk, state_t, bond_ind)
    E_ϕψ = E_local_ϕψ_Trotter(ising, rbm, rbm_t, Uk, state, bond_ind)
    #E_ψϕ = E_local_ψϕ_App(rbm, rbm_t, Uk, state_t, k)
    #E_ϕψ = E_local_ϕψ_App(rbm, rbm_t, Uk, state, k)
    #println("E_ψϕ[1] = $(E_ψϕ[1])")
    #println("E_ϕψ[1] = $(E_ϕψ[1])")
    #E_loc_t = -E_ψϕ .* E_ϕψ 
    ∂θ_t = Compute_derivatives_G(rbm_t.b[:], rbm_t.W[:,:], state_t, size(state_t,2), block_nm)
    #bond_ind = [k, rbm.nv_bond[k]]
    #∂θ_t = Compute_derivatives_G(rbm_t, state_t, N, bond_ind, bond_b)
    #E_loc_t1 =  (E_ψϕ)
    #E_loc_t2 =  (E_ϕψ)
    return E_ψϕ, E_ϕψ, ∂θ_t, E_ψϕ .* conj(∂θ_t), conj(E_ψϕ) .* conj(∂θ_t)
    #return E_loc_t1, E_loc_t2
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

function measure_G(quant_sys::T1, rbm_g::RBM, state::CuArray{Int,2}, ∂θ::CuArray{T2,2}, N::Int, nm::CuArray{Int,1}) where{T1 <: Abstract_Quantum_System, T2<:Number}
    E_loc = E_local_G(quant_sys, rbm_g, state, N, quant_sys.site_bond_y)
    ∂θ = Compute_derivatives_G(rbm_g.b[:],rbm_g.W[:,:], state, N, nm)
    E_loc, ∂θ, E_loc .* conj(∂θ) #∂θ just represents one portion of the whole derivatives of parameters.
end


# function Compute_derivatives_G(rbm_W::CuArray{T}, state::CuArray{Int,2}, E_loc_N :: CuArray{T,2}, N::Int, nm::Array{Int, 1}) where{T<:Number}  # Compute a part of ∂a and  ∂W
#     ∂a_nm = state[nm,:]
#     #cu_state = CuArray(state)
#     #∂a_nm = cu_state[nm,:]
#     #println("∂a= ",typeof(∂a))
#     C  = rbm_W * state
#     #∂b = (exp.(C).-exp.(-C))./(exp.(C).+exp.(-C))
#     #println("-----",typeof(∂b))
#     ∂b = tanh.(C)
#     ∂θ_nm = zeros(ComplexF64, size(∂b,1) * length(nm) + length(nm), N)
#     #S_kk_nm = zeros(ComplexF64, length(nm) + size(∂b,1) * length(nm), length(nm) + size(∂b,1) * length(nm))
#     for k = 1 : N
#         ∂W_nm_k = ∂b[:,k] * transpose(∂a_nm[:,k])
#         ∂θ_nm[:,k] = [∂a_nm[:,k] ; ∂W_nm_k[:]]
#         #S_kk_nm_cu = CuArray(conj(∂θ_nm[:,k])) * CuArray(transpose(∂θ_nm[:,k]))
#         #S_kk_nm .= S_kk_nm .+ conj(∂θ_nm[:,k]) * transpose(∂θ_nm[:,k])

#     end
#     S_kk_nm = conj(∂θ_nm) * transpose(∂θ_nm)
#     ∂θ_nm_avg = sum(∂θ_nm, dims = 2)
#     E_loc∂θ_nm = E_loc_N .* conj(∂θ_nm)
#     E_loc∂θ_nm_avg = sum(E_loc∂θ_nm, dims = 2)

#     #GC.gc()
#     return ∂θ_nm_avg[:],  E_loc∂θ_nm_avg[:], S_kk_nm
# end

function Compute_derivatives_G(rbm_b :: CuArray{T}, rbm_W::CuArray{T}, state::CuArray{Int,2}, N::Int, nm::CuArray{Int, 1}) where{T<:Number}  # Compute a part of ∂a, ∂b and ∂W
    ∂a_nm = state[nm,:]
    #println("∂a= ",typeof(∂a))
    C  = rbm_W * state .+ rbm_b
    ∂b = (exp.(C).-exp.(-C))./(exp.(C).+exp.(-C))
    #println("-----",typeof(∂b))
    #∂b = tanh.(C)
    ∂W = CuArray(zeros(T, size(∂b, 1)*size(∂a_nm, 1), N)) #∂θ is a column vector
    CUDA.@sync begin
        @cuda blocks = 128 threads = 512 Compute_∂W_nm!(∂W, ∂a_nm, ∂b, N)
    end
    #∂θ .= [∂a_nm; ∂W]
    #∂θ .= ∂W
    #return [∂a_nm; ∂W]
    return [∂a_nm; ∂b; ∂W]
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
	stride = blockDim().x #number of threads1
	stride_b = gridDim().x #number of blocks1
	n_∂θ = size(∂θ_nm,1) # number of parameters
	for i = ind_th_x:stride:n_∂θ
		for j = ind_x:stride_b:n_∂θ
			for n = 1:N_sampling
				S_kk[i,j] += conj(∂θ_nm[i,n]) * ∂θ_nm[j,n]
			end
		end
	end
	return nothing
end


#=======================#
#=======================#

########### --- Sampling --- ###########
########### --- Sampling --- ###########

mutable struct Sampler_Time <: Abstract_Sampling
    state_t::CuArray{Int,2}
end

function Sampler_Time(nv::Int, N::Int)
    state_t = CuArray(rand(-1:2:1,nv,N))
    return Sampler_Time(state_t)
end


mutable struct Sampling <: Abstract_Sampling
    n_thermals :: Int
    n_sweeps :: Int
    N_states :: Int #This means how many states we updates in one step(The states is a matrix, instead of an vector)
    counter :: Int
    state :: CuArray{Int,2}
end

mutable struct Importance_Sampling <: Abstract_Sampling
	importance_state :: Array{Int,2}
end

mutable struct Adam_Parameters
	m_t ::CuArray{ComplexF64, 1}
        v_t ::CuArray{ComplexF64, 2}
end

mutable struct Adam_opt
	m_a :: CuArray{ComplexF64, 1}
	m_W :: CuArray{ComplexF64, 2}
	v_a :: CuArray{ComplexF64, 1}
	v_W :: CuArray{ComplexF64, 2}
end

mutable struct rungle_kutta <: abstract_evolution
    order :: Int
    k :: Array{Int,1}
end

function Adam_Parameters(nv :: Int, nh :: Int)
	m_a = CuArray(zeros(ComplexF64,nv))
        m_W = CuArray(zeros(ComplexF64,nh,nv))
	v_a = CuArray(zeros(ComplexF64,nv))
	v_W = CuArray(zeros(ComplexF64,nh,nv))
	return Adam_opt(m_a, m_W, v_a, v_W)
end

function Adam_update!(adam::Adam_opt, rbm_t :: RBM_Time, ∂θ::CuArray{ComplexF64}, t :: CuArray{Int,1}, nm :: CuArray{Int}, nh :: Int; β₁ = 0.9, β₂ = 0.999, α=0.001, ϵ = 1e-8)
        ∂θ_W = reshape(∂θ[length(nm)+1 : end], (nh, length(nm)))
	adam.m_a[nm] .= β₁ * adam.m_a[nm] .+ (1 - β₁) * ∂θ[1:length(nm)]
	adam.v_a[nm] .= β₂ * adam.v_a[nm] .+ (1 - β₂) * (∂θ[1:length(nm)] .* ∂θ[1:length(nm)])

	adam.m_W[:,nm] .= β₁ * adam.m_W[:,nm] .+ (1 - β₁) * ∂θ_W
	adam.v_W[:,nm] .= β₂ * adam.v_W[:,nm] .+ (1 - β₂) * (∂θ_W .* ∂θ_W)

	m_a_t = adam.m_a[nm] ./ (1 .- β₁ .^ t[nm])
	v_a_t = adam.v_a[nm] ./ (1 .- β₂ .^ t[nm])
	v_a_sqrt = sqrt.(v_a_t)
	println("size of m_a_t = ",size(m_a_t))
	println("size of v_a_sqrt = ", size(v_a_sqrt))
	rbm_t.a[nm] .-= α * m_a_t ./ (v_a_sqrt .+ ϵ)

	m_W_t = adam.m_W[:,nm] ./ transpose(1 .- β₁ .^ t[nm])
	v_W_t = adam.v_W[:,nm] ./ transpose(1 .- β₂ .^ t[nm])
	v_W_sqrt = sqrt.(v_W_t)
	rbm_t.W[:,nm] .-= α * m_W_t ./ (v_W_sqrt .+ ϵ)
end
	





function Thermalization!(ising::Ising_Model, rbm ::T, s_g::Sampling, s_t::Sampler_Time) where{T<:Abstract_Learning_Machine}
    s_g.state = CuArray(rand(-1:2:1, ising.n_sites, s_g.N_states))
    s_t.state_t = CuArray(rand(-1:2:1, ising.n_sites, s_g.N_states))
    #sample.state[1:Int(ising.n_sites/2),:] .= 1
    #sample.state[Int(ising.n_sites/2) + 1:end,:] .= -1
    #for m = 1 : ising.n_sites
    #    sample.state[m,:] .= (-1)^(m+1)
    #end
    println("sample.state[:,1] = ",sample.state[:,1])
    
    for j = 1 : sample.n_thermals
        single_sweep_time!(ising, s_g, rbm)
        single_sweep_time!(ising, s_t, rbm_t, s_g)
    end
end

mutable struct Time_Evolution
    initial_time::Float64
    final_time::Float64
    dt::Float64
    time::Array{Float64}
end
function Time_Evolution(initial_time::Real, final_time::Real, dt::Real)
    time = collect(initial_time : dt : final_time)
    return Time_Evolution(initial_time, final_time, dt, time)
end

function single_sweep_time!(quant_sys::Ising_Model, s_g::Sampling, rbm:: RBM)
    L = quant_sys.n_sites
    N = s_g.N_states
    for j = 1 : L
        x_trial = copy(s_g.state)
        flip_local = CuArray(rand(1:L, N))
	#flip_local = CuArray(zeros(Int,N)) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        p = abs.(Ratio_ψ(rbm, s_g.state, flip_global)).^2
        #println("p= ",p)
        r = CUDA.rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("sr= "(,sr)
        s_ind = CuArray(findall(x->x==1,sr[:])) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        s_g.state[site_ind_accept] .= -s_g.state[site_ind_accept]
    end
end

function single_sweep_time!(quant_sys::Ising_Model, s_t::Sampler_Time, rbm_t::RBM_Time, s_g:: Sampling)
    L = quant_sys.n_sites
    N = s_g.N_states
    for j = 1 : L
        x_trial = copy(s_t.state_t)
        flip_local = CuArray(rand(1:L, N))
	#flip_local = CuArray(zeros(Int,N)) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        p = abs.(Ratio_ϕ(rbm_t, s_t.state_t, flip_global)).^2
        #println("p= ",p)
        r = CUDA.rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = CuArray(findall(x->x==1,sr[:])) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        s_t.state_t[site_ind_accept] .= -s_t.state_t[site_ind_accept]
    end

end

function single_sweep_time!(quant_sys::Ising_Model, s_g::Sampling, s_t::Sampler_Time, rbm:: RBM, rbm_t::RBM_Time)# Metroplis Algorithm to creat many samplers using GPU
    L = quant_sys.n_sites
    N = s_g.N_states
    for j = 1 : L
        x_trial = copy(s_g.state)
        flip_local = rand(1:L, N)
	#flip_local = zeros(Int,N) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        p = abs.(Ratio_ψ(rbm, s_g.state, flip_global)).^2
        #println("p= ",p)
        r = rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = findall(x->x==1,sr[:]) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        s_g.state[site_ind_accept] .= -s_g.state[site_ind_accept]
    end

    for j = 1 : L
        x_trial = copy(s_t.state_t)
        #flip_local = CuArray(rand(1:L, N))
	    flip_local = zeros(Int,N) .+ j 
        #println("flip_local = ", flip_local)
        flip_global = gen_flip_list(flip_local, L) # Global position of sites needed to flip 
        #println("flip_global = ", flip_global)
        #x_trial[flip_global[:,2]] .= -sample.state[flip_global[:,2]]; # flip these sites and generate new configurations.
        p = abs.(Ratio_ϕ(rbm_t, s_t.state_t, flip_global)).^2
        #println("p= ",p)
        r = rand(1,N) # generate random numbers 
        #println("r= ",r)
        sr = r.<min.(1,p) # Find the indics of configurations which accept the new configurations
        #println("sr= ",sr)
        s_ind = findall(x->x==1,sr[:]) # Find the indics of configurations which accept the new configurations
        site_ind_accept = flip_global[s_ind,2] # site is the global position of flipped spin, 
                                    # so it means that we choose which configuration can be accepted
        #println("site_ind_accept= ",site_ind_accept)                        
        s_t.state_t[site_ind_accept] .= -s_t.state_t[site_ind_accept]
    end

end

#=======================#
#=======================#

########### --- Main Code: Optimization --- ########### 
########### --- Main Code: Optimization --- ########### 

function run_time_SR_Trotter(Quantum_Model::T1, rbm::RBM{T2}, rbm_t:: RBM_Time, sample::Sampling,  s_t::Sampler_Time, Uk::Un, block_nm::Any, ham_k ::CuArray{Int,1}) where{T1 <: Abstract_Quantum_System, T2 <: ComplexF64} # update all parameters in parallel
	if block_nm == "all"
	    #l_nm = rbm._nv + rbm._nv * rbm._nh #updat all parameters
	    l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters 
		elseif typeof(block_nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
			#l_nm = length(nm)*rbm._nh
			#1println("------------",typeof(block_nm))
			#l_nm = length(block_nm) + length(block_nm) * rbm._nh # update a and W
		        l_nm = length(block_nm) + rbm._nh + length(block_nm) * rbm._nh # update a, b and W
		else
			println("------------",typeof(block_nm))
			error("The setting of Block is wrong!!!")
    end
    #println("nh = ", rbm._nh)
    #println("l_nm = $l_nm ") 
    N = sample.N_states
    E_loc_avg = 0.0+0.0im
    E_loc_ψϕ_avg = 0.0+0.0im
    E_loc_ϕψ_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    #E_loc∂θ_conj_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm))
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    #∂θ_initial = CuArray(zeros(T2,l_nm, sample.N_states))
    #println("ham_k = ",ham_k)

    #for k = 1 : n_blocks
    #    l_nm = length(block_nm[k]) + length(block_nm[k]) * rbm._nh
    #    ∂θ_avg[k] = zeros(T2, l_nm)
    #    E_loc∂θ_avg[k] = zeros(T2, l_nm)
    #    ∂θ_Matrix[k] = zeros(T2, l_nm, l_nm)
    #end
    #block_nm_c = CuArray(block_nm)
    #println(size(∂θ_avg[1]))
    tot_time_S = 0.0
    tot_time_e_loc = 0.0
    #println("rbm.a[1:5] = ",rbm.a[1:5])
    #println("rbm_t.a[1:5] = ",rbm_t.a[1:5])
    #println("learning rate γ = ", rbm.γ)

    x_t1 = CuArray(ones(Int, Quantum_Model.n_sites, 2))
    x_t1[:,2] .= -1
    #println("Test congfiguration is ", x_t1)
    R_t1 = Ratio_ψϕ(rbm, rbm_t, x_t1, x_t1)
    R_t2 = Ratio_ϕψ(rbm_t, rbm, x_t1, x_t1)
    #println("R_t1 = ", R_t1)
    #println("R_t2 = ", R_t2)
    sum_R_t1 = Ratio_ψϕ(rbm, rbm_t, s_t.state_t[:,:],s_t.state_t[:,:])
    sum_R_t2 = Ratio_ϕψ(rbm_t, rbm, s_t.state_t[:,:],s_t.state_t[:,:])
    sum_R_t1 = sum(sum_R_t1)
    sum_R_t2 = sum(sum_R_t2)
    #println("sum R_t1 = ", sum_R_t1)
    #println("sum R_t2 = ", sum_R_t2)

    io4 = open("Ratio_E_loc_$(rbm._nv)_sites.txt", "a+")
    println(io4, sum_R_t2)
    close(io4)
    
    tot_time_S = @elapsed for j = 1 : sample.n_sweeps 
        single_sweep_time!(Quantum_Model, sample, rbm)
        single_sweep_time!(Quantum_Model, s_t, rbm_t, sample)
        #s_t.state_t[:,:] .= sample.state[:,:]
        #E_loc, ∂θ_nm, E_loc∂θ = measure_time_SR_Trotter(Quantum_Model, rbm, rbm_t, Uk, sample.state[:,:], s_t.state_t[:,:], ham_k,block_nm)
            
        #E_loc_avg += sum(E_loc)
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:, site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
	
	    E_ψϕ, E_ϕψ, ∂θ_nm, E_loc∂θ, E_loc∂θ_conj =  measure_time_SR_Trotter_New(Quantum_Model, rbm, rbm_t, Uk, sample.state[:,:], s_t.state_t[:,:], ham_k,block_nm)
        #println("E_ψϕ[1] = ",E_ψϕ[1])
        #println("E_ϕψ[1] = ",E_ϕψ[1])
        #E_ϕψ = real(E_ϕψ) - imag(E_ψϕ) * im
        #E_ϕψ = conj(E_ψϕ)
        E_loc_avg += -1 * sum(E_ψϕ .* E_ϕψ)
        E_loc_ψϕ_avg += sum(E_ψϕ)
        E_loc_ϕψ_avg += sum(E_ϕψ)
        ∂θ_avg += sum(∂θ_nm,dims = 2)
        E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
        #E_loc∂θ_conj_avg += sum(E_loc∂θ_conj, dims = 2)


        S_kk .= 0
        #CUDA.@sync begin
        #	@cuda  blocks = 8192 threads = 1024  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, sample.N_states);		 
        #end
        ∂θ_Matrix .+= conj(∂θ_nm) * transpose(∂θ_nm)
        #∂θ_avg += sum(∂θ_nm,dims = 2)
        #E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
        #tot_time_e_loc += tot_time_e_loc_m
        # compute derivatives of parameters in parallel
        #println("if it works.............")
        
        # println("It works too!!!!!")
        #println(size(∂θ_avg[2]))
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:,site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
    end
    #E_avg_new = E_loc_ψϕ_avg/()
    #println("New Overlap = ")
    #println("Time of sampling === $tot_time_S")

    #println("Time of measuring = $tot_time_e_loc")
    # for j = 1 : s_g.n_sweeps
    #     for m = 1:s_g.rbm_g._nv
    #         single_sweep_time!(s_g, s_t, rbm_t)
    #     end
    #     #s_g.state[:,:] .= 1
    #     #s_t.state_t[:,:] .= 1

    #     E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, nm, N)
    #     #E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, N)
    #     E_loc_avg += sum(E_loc_t)
    #     S_kk .=0
    #     CUDA.@sync begin
    #         @cuda  blocks = 100 threads = 500  Compute_∂θ_Matrix!(S_kk, ∂θ_nm_t, N);
    #     end
    #     ∂θ_avg += sum(∂θ_nm_t,dims = 2)
    #     ∂θ_Matrix .+= S_kk
    #     E_loc∂θ_avg += sum(E_loc∂θ_t,dims=2)
    # end
    N_total = sample.n_sweeps * sample.N_states
    
    E_loc_ψϕ_avg /= N_total
    E_loc_ϕψ_avg /= N_total
    E_avg_new = abs(-E_loc_ψϕ_avg * E_loc_ϕψ_avg + 1)
    #println("E_loc_ψϕ_avg = ", E_loc_ψϕ_avg)
    #println("E_loc_ϕψ_avg = ", E_loc_ϕψ_avg)
    E_loc_avg /= N_total
    #println("Old overlap   = ", 1 + E_loc_avg)
    #println("New overlap   = $E_avg_new")
    #println("New overlap 2 = ", -abs(E_loc_ψϕ_avg * E_loc_ϕψ_avg) + 1)
    #E_loc_avg /= N_total
    E_loc_avg = -E_loc_ψϕ_avg * E_loc_ϕψ_avg
    ∂θ_avg /= N_total
    E_loc∂θ_avg ./= N_total
    #E_loc∂θ_conj_avg ./= N_total
    #println("E_loc_avg = ", E_loc_avg)
    #E_loc_avg /=  (N_total) 
    #∂θ_avg ./= N_total
    #E_loc∂θ_avg ./= N_total
    ∂θ_Matrix ./= N_total
    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    #σ_avg /= s.n_sweeps
    #σz_avg /= s.n_sweeps
    
    #S = compute_Skl_GS(∂θ_list)
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
   # println("type of S =   ", typeof(S))
    S_r = compute_sr_gradient_G_gpu_S(S, sample.counter)
    S = nothing
    GC.gc()
    CUDA.reclaim()
    #gradient_force = E_loc∂θ_avg * E_loc_ϕψ_avg + E_loc∂θ_conj_avg * conj(E_loc_ϕψ_avg) - 2 * E_loc_ψϕ_avg * E_loc_ϕψ_avg * conj(∂θ_avg)
    gradient_force = -E_loc∂θ_avg * E_loc_ϕψ_avg - E_loc_avg * conj(∂θ_avg)
    #gradient_force = -1 * gradient_force
    gNQS = S_r \ gradient_force
    #gradient_force = 2 * real.(E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg)) .+ 0.0im
    #gNQS = S_r \ (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #gNQS = (1*(E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg)))
    #S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg) 
    #println("type of S =   ", typeof(S))
    #Sinv = pinv(S)
    #Sinv = compute_sr_gradient_G_gpu(S,s_g)
    #Sinv .= Sinv.^(-1)
    #Sinv = compute_sr_gradient(∂θ_list, s)
    #gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    #gNQS = (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #E_loc_avg, σ_avg, σz_avg, Array(gNQS)
    abs(E_loc_avg + 1),  gNQS[:]
    #-abs(E_loc_avg) + 1, gNQS[:]
end



function run_time_SR_Trotter(Quantum_Model::T1, rbm::RBM{T2}, rbm_t:: RBM_Time, sample::Sampling,  s_t::Sampler_Time, Uk::Un, block_nm::Any, ham_k ::CuArray{Int,1}, state_saved :: CuArray{Int,2}, state_t_saved :: CuArray{Int,2}) where{T1 <: Abstract_Quantum_System, T2 <: ComplexF64, T3 <: Importance_Sampling} # update all parameters in parallel
	if block_nm == "all"
	    l_nm = rbm._nv + rbm._nv * rbm._nh #updat all parameters
	    #l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters 
		elseif typeof(block_nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
			#l_nm = length(nm)*rbm._nh
			l_nm = length(block_nm) + length(block_nm) * rbm._nh # update a 	and W
		else
			error("The setting of Block is wrong!!!")
    end
    #println("l_nm = $l_nm ") 
    N = sample.N_states
    E_loc_avg = 0.0+0.0im
    E_loc_ψϕ_avg = 0.0+0.0im
    E_loc_ϕψ_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    #E_loc∂θ_conj_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm))
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    #∂θ_initial = CuArray(zeros(T2,l_nm, sample.N_states))
    #println("ham_k = ",ham_k)

    #for k = 1 : n_blocks
    #    l_nm = length(block_nm[k]) + length(block_nm[k]) * rbm._nh
    #    ∂θ_avg[k] = zeros(T2, l_nm)
    #    E_loc∂θ_avg[k] = zeros(T2, l_nm)
    #    ∂θ_Matrix[k] = zeros(T2, l_nm, l_nm)
    #end
    #block_nm_c = CuArray(block_nm)
    #println(size(∂θ_avg[1]))
    tot_time_S = 0.0
    tot_time_e_loc = 0.0
    #println("rbm.a[1:5] = ",rbm.a[1:5])
    #println("rbm_t.a[1:5] = ",rbm_t.a[1:5])
    #println("learning rate γ = ", rbm.γ)
    #tot_time_S = @elapsed for j = 1 : sample.n_sweeps 
        #single_sweep_time!(Quantum_Model, sample, rbm)
        #single_sweep_time!(Quantum_Model, s_t, rbm_t, sample)
	#s_t.state_t[:,:] .= sample.state[:,:]
    E_loc, ∂θ_nm, E_loc∂θ = measure_time_SR_Trotter(Quantum_Model, rbm, rbm_t, Uk, state_saved[:,:], state_t_saved[:,:], ham_k,block_nm)
    E_loc_avg += sum(E_loc)
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:, site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
	
	#E_ψϕ, E_ϕψ, ∂θ_nm, E_loc∂θ, E_loc∂θ_conj =  measure_time_SR_Trotter_New(Quantum_Model, rbm, rbm_t, Uk, sample.state[:,:], s_t.state_t[:,:], ham_k,block_nm)

	#E_loc_ψϕ_avg += sum(E_ψϕ)
	#E_loc_ϕψ_avg += sum(E_ϕψ)
    ∂θ_avg += sum(∂θ_nm,dims = 2)
    E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
	#E_loc∂θ_conj_avg += sum(E_loc∂θ_conj, dims = 2)


    S_kk .= 0
	#CUDA.@sync begin
	#	@cuda  blocks = 8192 threads = 1024  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, sample.N_states);		 
	#end
    ∂θ_Matrix .+= conj(∂θ_nm) * transpose(∂θ_nm)
        #∂θ_avg += sum(∂θ_nm,dims = 2)
        #E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
        #tot_time_e_loc += tot_time_e_loc_m
        # compute derivatives of parameters in parallel
        #println("if it works.............")
        
        # println("It works too!!!!!")
        #println(size(∂θ_avg[2]))
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:,site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
    #end

    #println("Time of sampling === $tot_time_S")

    #println("Time of measuring = $tot_time_e_loc")
    # for j = 1 : s_g.n_sweeps
    #     for m = 1:s_g.rbm_g._nv
    #         single_sweep_time!(s_g, s_t, rbm_t)
    #     end
    #     #s_g.state[:,:] .= 1
    #     #s_t.state_t[:,:] .= 1

    #     E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, nm, N)
    #     #E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, N)
    #     E_loc_avg += sum(E_loc_t)
    #     S_kk .=0
    #     CUDA.@sync begin
    #         @cuda  blocks = 100 threads = 500  Compute_∂θ_Matrix!(S_kk, ∂θ_nm_t, N);
    #     end
    #     ∂θ_avg += sum(∂θ_nm_t,dims = 2)
    #     ∂θ_Matrix .+= S_kk
    #     E_loc∂θ_avg += sum(E_loc∂θ_t,dims=2)
    # end
    N_total = sample.n_sweeps * sample.N_states
    
    #E_loc_ψϕ_avg /= N_total
    #E_loc_ϕψ_avg /= N_total
    E_loc_avg /= N_total	
    ∂θ_avg /= N_total
    E_loc∂θ_avg ./= N_total
    #E_loc∂θ_conj_avg ./= N_total
    #println("E_loc_avg = ", E_loc_avg)
    #E_loc_avg /=  (N_total) 
    #∂θ_avg ./= N_total
    #E_loc∂θ_avg ./= N_total
    ∂θ_Matrix ./= N_total
    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    #σ_avg /= s.n_sweeps
    #σz_avg /= s.n_sweeps
    
    #S = compute_Skl_GS(∂θ_list)
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
   # println("type of S =   ", typeof(S))
    S_r = compute_sr_gradient_G_gpu_S(S, sample.counter)
    S = nothing
    GC.gc()
    CUDA.reclaim()
    #gradient_force = E_loc∂θ_avg * E_loc_ϕψ_avg + E_loc∂θ_conj_avg * conj(E_loc_ϕψ_avg) - 2 * E_loc_ψϕ_avg * E_loc_ϕψ_avg * conj(∂θ_avg)
    #gradient_force = E_loc∂θ_avg * E_loc_ϕψ_avg - E_loc_ψϕ_avg * E_loc_ϕψ_avg * conj(∂θ_avg)
    #gradient_force = -1 * gradient_force
    #gNQS = S_r \ gradient_force
    gNQS = S_r \ (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #gNQS = (1*(E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg)))
    #S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg) 
    #println("type of S =   ", typeof(S))
    #Sinv = pinv(S)
    #Sinv = compute_sr_gradient_G_gpu(S,s_g)
    #Sinv .= Sinv.^(-1)
    #Sinv = compute_sr_gradient(∂θ_list, s)
    #gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    #gNQS = (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #E_loc_avg, σ_avg, σz_avg, Array(gNQS)
    abs(E_loc_avg + 1),  gNQS[:]
end





function run_time_SR_Trotter_IS(Quantum_Model::T1, rbm::RBM{T2}, rbm_t:: RBM_Time, sample::Sampling,  s_t::Sampler_Time, Uk::Un, block_nm::Any, ham_k ::CuArray{Int,1}, tol :: Number) where{T1 <: Abstract_Quantum_System, T2 <: ComplexF64} # update all parameters in parallel
	if block_nm == "all"
	    l_nm = rbm._nv + rbm._nv * rbm._nh #updat all parameters
	    #l_nm = rbm._nv + rbm._nh + rbm._nv * rbm._nh #updat all parameters 
		elseif typeof(block_nm) == CuArray{Int,1,CUDA.Mem.DeviceBuffer}
			#l_nm = length(nm)*rbm._nh
			l_nm = length(block_nm) + length(block_nm) * rbm._nh # update a 	and W
		else
			error("The setting of Block is wrong!!!")
    end
    println("l_nm = $l_nm ") 
    N = sample.N_states
    E_loc_avg = 0.0+0.0im
    E_loc_ψϕ_avg = 0.0+0.0im
    E_loc_ϕψ_avg = 0.0+0.0im
    ∂θ_avg = CuArray(zeros(T2,l_nm))
    E_loc∂θ_avg = CuArray(zeros(T2,l_nm))
    #E_loc∂θ_conj_avg = CuArray(zeros(T2,l_nm))
    ∂θ_Matrix = CuArray(zeros(T2, l_nm, l_nm))
    S_kk = CuArray(zeros(T2,l_nm,l_nm))
    #∂θ_initial = CuArray(zeros(T2,l_nm, sample.N_states))
    #println("ham_k = ",ham_k)

    #for k = 1 : n_blocks
    #    l_nm = length(block_nm[k]) + length(block_nm[k]) * rbm._nh
    #    ∂θ_avg[k] = zeros(T2, l_nm)
    #    E_loc∂θ_avg[k] = zeros(T2, l_nm)
    #    ∂θ_Matrix[k] = zeros(T2, l_nm, l_nm)
    #end
    #block_nm_c = CuArray(block_nm)
    #println(size(∂θ_avg[1]))
    tot_time_S = 0.0
    tot_time_e_loc = 0.0
    #println("rbm.a[1:5] = ",rbm.a[1:5])
    #println("rbm_t.a[1:5] = ",rbm_t.a[1:5])
    #println("learning rate γ = ", rbm.γ)

    N_total = sample.n_sweeps * sample.N_states
    IS_sample = CuArray(zeros(Int, Quantum_Model.n_sites, N_total ))
    IS_sample_t = CuArray(zeros(Int, Quantum_Model.n_sites, N_total ))


    tot_time_S = @elapsed for j = 1 : sample.n_sweeps 
        single_sweep_time!(Quantum_Model, sample, rbm)
        single_sweep_time!(Quantum_Model, s_t, rbm_t, sample)

	IS_sample[:, (j - 1) * sample.N_states + 1 : j * sample.N_states] .= sample.state[:,:]
	IS_sample_t[:, (j - 1) * sample.N_states + 1 : j * sample.N_states] .= s_t.state_t[:,:]

	#s_t.state_t[:,:] .= sample.state[:,:]
	#E_loc, ∂θ_nm, E_loc∂θ = measure_time_SR_Trotter(Quantum_Model, rbm, rbm_t, Uk, sample.state[:,:], s_t.state_t[:,:], ham_k,block_nm)
        #E_loc_avg += sum(E_loc)
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:, site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
	
	#E_ψϕ, E_ϕψ, ∂θ_nm, E_loc∂θ, E_loc∂θ_conj =  measure_time_SR_Trotter_New(Quantum_Model, rbm, rbm_t, Uk, sample.state[:,:], s_t.state_t[:,:], ham_k,block_nm)

	#E_loc_ψϕ_avg += sum(E_ψϕ)
	#E_loc_ϕψ_avg += sum(E_ϕψ)
	#∂θ_avg += sum(∂θ_nm,dims = 2)
	#E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
	#E_loc∂θ_conj_avg += sum(E_loc∂θ_conj, dims = 2)


        #S_kk .= 0
	#CUDA.@sync begin
	#	@cuda  blocks = 8192 threads = 1024  Compute_∂θ_Matrix!(S_kk, ∂θ_nm, sample.N_states);		 
	#end
        #∂θ_Matrix .+= conj(∂θ_nm) * transpose(∂θ_nm)
        #∂θ_avg += sum(∂θ_nm,dims = 2)
        #E_loc∂θ_avg += sum(E_loc∂θ,dims=3)
        #tot_time_e_loc += tot_time_e_loc_m
        # compute derivatives of parameters in parallel
        #println("if it works.............")
        
        # println("It works too!!!!!")
        #println(size(∂θ_avg[2]))
        # for site_j = 1 : Quantum_Model.n_sites
        #     col_avg[:,site_j] = col_avg[:,site_j] + sum(sample.state[site_j,:]' .* sample.state, dims = 2)
        # end
    end

    #println("Time of sampling === $tot_time_S")

    rbm_n = RBM_n(rbm_t.a[:], rbm_t.W[:,:])

    #E_ψϕ = E_local_ψϕ_Trotter(Quantum_Model, rbm, rbm_t, Uk, IS_sample_t, ham_k)
    #E_ϕψ = E_local_ϕψ_Trotter(Quantum_Model, rbm, rbm_t, Uk, IS_sample, ham_k)
    #∂θ_nm = Compute_derivatives_G(rbm_t.W[:,:], IS_sample_t, size(IS_sample_t , 2), block_nm)

    E_loc_avg_new  = 0.0 + 0.0im
    ϵ = 0.05
    for iter_k = 1 : 100
	    R_n = abs.(Ratio_ϕ(rbm_t, rbm_n, IS_sample_t)).^2
	    sum_weight = sum(R_n)
	    #println("sum_weight = ", sum_weight / N_total)
	    if abs(sum_weight) / N_total < (1 - ϵ) || abs(sum_weight) / N_total > (1 + ϵ)
		    println("Tis is due to the weight exceed the truncation")
		    println("Count = " , iter_k)
		    break
	    end
 
 	    E_ψϕ = E_local_ψϕ_Trotter(Quantum_Model, rbm, rbm_n, Uk, IS_sample_t, ham_k)
	    E_ϕψ = E_local_ϕψ_Trotter(Quantum_Model, rbm, rbm_n, Uk, IS_sample, ham_k)
	    #∂θ_nm = Compute_derivatives_G(rbm_n.W[:,:], IS_sample_t, size(IS_sample_t , 2), block_nm)

	    #E_loc_avg = -1 * sum(R_n .* E_ψϕ .* E_ϕψ) / sum_weight
	    E_loc_avg = -1 * sum(R_n .* E_ψϕ) * sum(E_ϕψ) / (sum_weight * N_total)
	    #E_loc_avg_new = -1 * sum(R_n .* E_ψϕ) * sum(E_ϕψ) / (sum_weight * N_total) + 1
	    E_loc_avg_new = abs(E_loc_avg + 1)
	    if abs(E_loc_avg_new) < tol
		    println("Count = " , iter_k)
		    break
	    end
	    ∂θ_nm = Compute_derivatives_G(rbm_n.b[:], rbm_n.W[:,:], IS_sample_t, size(IS_sample_t , 2), block_nm)

	    ∂θ_avg = sum(R_n .* ∂θ_nm, dims = 2) / sum_weight
	    #E_loc∂θ_avg = -1 * sum(R_n .* E_ψϕ .* E_ϕψ .* conj(∂θ_nm), dims = 2) / sum_weight
	    E_loc∂θ_avg = -1 * sum(R_n .* E_ψϕ .* conj(∂θ_nm) , dims = 2) * sum(E_ϕψ)/ (sum_weight * N_total)
	    ∂θ_Matrix = (R_n .* conj(∂θ_nm)) * transpose(∂θ_nm) / sum_weight

	    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
	    S_r = compute_sr_gradient_G_gpu_S(S, sample.counter)
	    #grad_force = -sum(R_n .* E_ψϕ .* conj(∂θ_nm), dims = 2) .* sum(E_ϕψ) / (sum_weight * N_total) .-  sum(R_n .* E_ψϕ) * sum(R_n .* E_ϕψ .* conj(∂θ_nm), dims = 2) / sum_weight^2 .- 2 * E_loc_avg .* sum(R_n .* conj(∂θ_nm), dims = 2) / sum_weight 
	    gNQS = S_r \ (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
            #gNQS = S_r \ grad_force
	    update_parameters_block_IS!(rbm_n, rbm, gNQS[:], block_nm)
    end

    rbm_t.a[:] .= rbm_n.a[:]
    rbm_t.W[:,:] .= rbm_n.W[:,:]
    println("=====>>>>>  (After) Overleap = ", E_loc_avg_new)








    #println("Time of measuring = $tot_time_e_loc")
    # for j = 1 : s_g.n_sweeps
    #     for m = 1:s_g.rbm_g._nv
    #         single_sweep_time!(s_g, s_t, rbm_t)
    #     end
    #     #s_g.state[:,:] .= 1
    #     #s_t.state_t[:,:] .= 1

    #     E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, nm, N)
    #     #E_loc_t, ∂θ_nm_t, E_loc∂θ_t  = measure_time_SR_Trotter(s_g.rbm_g, rbm_t, Uk, s_g.state, s_t.state_t, k, ∂θ_initial, N)
    #     E_loc_avg += sum(E_loc_t)
    #     S_kk .=0
    #     CUDA.@sync begin
    #         @cuda  blocks = 100 threads = 500  Compute_∂θ_Matrix!(S_kk, ∂θ_nm_t, N);
    #     end
    #     ∂θ_avg += sum(∂θ_nm_t,dims = 2)
    #     ∂θ_Matrix .+= S_kk
    #     E_loc∂θ_avg += sum(E_loc∂θ_t,dims=2)
    # end
    #N_total = sample.n_sweeps * sample.N_states
    
    #E_loc_ψϕ_avg /= N_total
    #E_loc_ϕψ_avg /= N_total
    #E_loc_avg /= N_total	
    #∂θ_avg /= N_total
    #E_loc∂θ_avg ./= N_total
    #E_loc∂θ_conj_avg ./= N_total
    #println("E_loc_avg = ", E_loc_avg)
    #E_loc_avg /=  (N_total) 
    #∂θ_avg ./= N_total
    #E_loc∂θ_avg ./= N_total
    #∂θ_Matrix ./= N_total
    #S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    #σ_avg /= s.n_sweeps
    #σz_avg /= s.n_sweeps
    
    #S = compute_Skl_GS(∂θ_list)
    ∂θ_Matrix = nothing
    S_kk = nothing
    ∂θ_initial = nothing
   # println("type of S =   ", typeof(S))
    #S_r = compute_sr_gradient_G_gpu_S(S, sample.counter)
    #S = nothing
    GC.gc()
    CUDA.reclaim()
    #gradient_force = E_loc∂θ_avg * E_loc_ϕψ_avg + E_loc∂θ_conj_avg * conj(E_loc_ϕψ_avg) - 3 * E_loc_ψϕ_avg * E_loc_ϕψ_avg * conj(∂θ_avg)
    #gradient_force = E_loc∂θ_avg * E_loc_ϕψ_avg - E_loc_ψϕ_avg * E_loc_ϕψ_avg * conj(∂θ_avg)


    #gradient_force = -1 * gradient_force
    #gNQS = S_r \ gradient_force
    #gNQS = S_r \ (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #gNQS = (1*(E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg)))
    #S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg) 
    #println("type of S =   ", typeof(S))
    #Sinv = pinv(S)
    #Sinv = compute_sr_gradient_G_gpu(S,s_g)
    #Sinv .= Sinv.^(-1)
    #Sinv = compute_sr_gradient(∂θ_list, s)
    #gNQS = Sinv * (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))

    #gNQS = (E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg))
    #E_loc_avg, σ_avg, σz_avg, Array(gNQS)
    abs(E_loc_avg_new)
end

function optimize_params(quant_sys::T_model, rbm::RBM, sample::Sampling,
    n_params::Int, rk::rungle_kutta)where{T_model<:Abstract_Quantum_System}
    E_loc_avg = 0.0+0.0im
    T2 = ComplexF64
    ∂θ_avg = CuArray(zeros(T2,n_params))
    E_loc∂θ_avg = CuArray(zeros(T2,n_params))
    ∂θ_Matrix = CuArray(zeros(T2, n_params, n_params))
    ∂θ = CUDA.zeros(ComplexF64, n_params, sample.N_states)
    for j = 1 : sample.n_sweeps
        #∂θ = calculate_derivatives!(nn, sample.state, sample.n_states, ∂θ, "all", "all", "all")
        E_loc, ∂θ, E_loc∂θ  = measure_G(quant_sys, rbm, sample.state, ∂θ, sample.N_states, CuArray(collect(1 : rbm._nv)))
        E_loc_avg += sum(E_loc)
        ∂θ_avg += sum(∂θ,dims = 2)
        E_loc∂θ_avg += sum(E_loc∂θ,dims=2)
        ∂θ_Matrix .+= conj(∂θ) * transpose(∂θ)
    end
    #exact_ψ = exp.(activation_func(act_func, nn.b, nn.W, sample.state, nn.n_layers - 1))
    #exact_ψ = exp.(nn.model(sample.state))
    #P_ψ = abs.(exact_ψ).^2 / sum(abs.(exact_ψ).^2)
    #accum_opt_params!(rk, opt_type, opt_params_avg, E_loc, ∂θ, P_ψ)
    n_total = sample.n_sweeps * sample.N_states
    E_loc_avg /= n_total
    ∂θ_avg ./= n_total
    E_loc∂θ_avg ./= n_total
    ∂θ_Matrix  ./= n_total
    S = ∂θ_Matrix .- conj(∂θ_avg) * transpose(∂θ_avg)
    ∂θ_Matrix = nothing
    gradient_force = E_loc∂θ_avg .- E_loc_avg * conj.(∂θ_avg)
    gNQS = -im * (S \ gradient_force)
    S = nothing
    #GC.gc()
    #CUDA.reclaim() 
    ∂θ = nothing
    return E_loc_avg, gNQS[:]
end

function compute_sr_gradient_G_gpu(S::CuArray{T, 2}, counter :: Int ) where T
    S = regularize_G(S, counter)
    S += 1e-8 * I 
    Sinv = inv(S) 
    return Sinv
end

function compute_sr_gradient_G_gpu_S(S::CuArray{T, 2}, counter :: Int ) where T
    S = regularize_G(S, counter)
    S += 1e-8 * I 
    return S
end



function regularize_G(S::CuArray{T, 2}, counter::Int; λ₀=100, b=0.9, λ_min = 1e-4) where T
    p = counter
    λ = x -> max(λ₀ * b^p, λ_min)
    Sreg = S + λ(p) * Diagonal(diag(S))
end

function SR_updates_SF(∂θ_avg_nm::CuArray{T, 1}, E_loc_avg::T, E_loc∂θ_avg_nm::CuArray{T,1}, ∂θ_Matrix_nm::Array{T,2}, N_total::Int, counter :: Int) where{T<:Number}
    Sinv_nm = compute_sr_gradient_G_gpu(∂θ_Matrix_nm, counter)
    gNQS_nm = Sinv_nm * (E_loc∂θ_avg_nm .- E_loc_avg * conj.(∂θ_avg_nm))
    return gNQS_nm

end


function optimize_t_Trotter_time_Local_block(Quantum_Model::T1, Lx::Int, Ly::Int, rbm::RBM, block_size_xy::Array{Int,1}, num_epochs::Int, n_loop::Int, s_g::Sampling,  evo::Time_Evolution,  tol::Number, update_mode::String, LR_coff::Float64, final_γ :: Float64, segment_redu_γ::Int, random_seed_num::Int, N_Ham_bond :: Int, Sample_Mode :: String) where{T1<:Abstract_Quantum_System, T2<:Abstract_Learning_Machine}
    
    # divide parameters into blocks 
    Block_ind = []
    Block_tot = []
    block_size = block_size_xy[1] 
    if update_mode == "all"
	    push!(Block_ind, CuArray(collect(1 : Quantum_Model.n_sites)))
        N_blocks = 1
        println("All parameters will be updated")
    elseif update_mode == "block"
            if Ly == 1 # 1D case
                    #push!(Block_ind, 0) # generate the index of Block, the first index is 0, which means a and b parameters are updated.
                    
		    N_blocks = ceil(Int,Quantum_Model.n_sites/block_size) # block_size means how many sites in every block
              	    #if mod(Lx,2) != 0
		    #	    N_blocks -= 1
		    #end
		    #        for j = 1 : N_blocks
            #            j < N_blocks ? push!(Block_ind,CuArray(collect((j-1)*block_size+1:j*block_size))) : push!(Block_ind,CuArray(collect((j-1)*block_size+1:Quantum_Model.n_sites)))
            #         end
                    for j = 1 : 2 * N_blocks - 1
                            block_size_2 = Int(block_size/2)
                            push!(Block_ind, CuArray(collect( (j-1)*block_size_2 + 1 : block_size_2 * (j+1)  )))
                    end
		    if mod(Lx ,2) != 0
			    Block_ind = Block_ind[1:end-1]
		    end
            elseif Ly != 1 # 2D case
                    if block_size_xy == [2, 1]
                            Block_ind = generate_block_2D(Quantum_Model)
                    else
                            Block_ind = generate_block_2D(Lx, Ly, block_size_xy)    
                    end
            end
	    Block_tot = Block_ind
        Block_ind = [Block_ind ; reverse(Block_ind)]
        pop!(Block_ind) #remove the first block from Block index
        deleteat!(Block_ind, Int((length(Block_ind) + 1)*0.5)) #remove the last block from Block index
        #println("size of Block = ", Int(size(Block_ind,1)/2))
        println("number of Block = ", Int(size(Block_ind,1)))
        else
        error("There isn't such option: $update_mode. The option of update mode are: all and block.")
    end

    #Ham_ind = collect(1 : N_Ham_bond : Quantum_Model.n_sites - N_Ham_bond)
    Ham_ind = collect(1 : N_Ham_bond : Quantum_Model.n_sites )
    if Ham_ind[end] == Quantum_Model.n_sites
        Ham_ind = Ham_ind[1:end-1]
    end

    Ham_ind = [Ham_ind; reverse(Ham_ind)]
    #deleteat!(Ham_ind, Int(length(Ham_ind)/2))
    println("The Ham_ind = ", Ham_ind)
    
    Block_ind_0 = Block_ind


    println("The block = ", Block_ind)
    println("The Block_tot = ", Block_tot)

    Total_sample = s_g.n_sweeps * s_g.N_states
    
    tn = length(evo.time)
    E_t = zeros(ComplexF64, tn)
    #res = zeros(num_epochs).+0.0im
    nv = rbm._nv
    nh = rbm._nh
    α_h = Int(nh/nv)
    
    #Ham_ind = [collect(1:nv-1);collect(nv-1:-1:1)]
    #Ham_ind = [collect(1:2:nv-1);collect(2:2:nv);collect(nv:-2:2);collect(nv-1:-2:1)]
    


    
    #println(Block_ind)
    #rbm_t = RBM_Time(s_g.rbm_g.a, s_g.rbm_g.b, s_g.rbm_g.W)
    s_t=Sampler_Time(Quantum_Model.n_sites , s_g.N_states)

    s_t.state_t[:,:] .= s_g.state[:,:]

    #Uk = Un(Quantum_Model, evo.dt, 1)
    total_s = s_g.N_states*s_g.n_sweeps
    #io = open("Time_Evolution_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "w");
    #io = open("Time_Evolution_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+");
    #println(io, "Date: ",Dates.Time(Dates.now()), "    Tol = ",tol, "   #Sampling = ",s_g.n_sweeps*s_g.n_state, "  hx = ",s_g.rbm_g.h, "  hz = ", s_g.rbm_g.hz, "  hx_t = ",hx_t)
    #close(io)

    io2 = open("Overlap_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "w");
    io2 = open("Overlap_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+");

    io3 = open("sum_NN_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "w")
    io3 = open("sum_NN_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+")

    io4 = open("Ratio_E_loc_$(nv)_sites.txt", "w")
    io4 = open("Ratio_E_loc_$(nv)_sites.txt", "a+")

    Overlap_t = 0
    γ₀ = rbm.γ
    println("Ready to run time evolution !!! ============")

    for t_ind = 2 : tn
        println("-----------")
        Ham_counter = 0
        count_k = 1
        for k in Ham_ind
            #for k = Ham_ind
            #k = Ham_ind[count_k]
            Ham_counter += 1
            s_g.counter = 0

            adam_counter = CuArray(zeros(Int, Quantum_Model.n_sites))
            adam_para = Adam_Parameters(rbm._nv, rbm._nh)

            println("--->  k = ",k)
            #site_k = CuArray(collect(k:k+N_Ham_bond))
            site_k = CuArray(collect(k : min(k + N_Ham_bond, Quantum_Model.n_sites)))
            println("site_k = ", site_k)
            Loop = 1
            Uk = Un_m(Quantum_Model, evo.dt, k, N_Ham_bond)
            Block_ind = circshift(Block_ind_0, -(k-1))
            println("Block_ind_k = ", Block_ind)
            println("length of block_tot = ", length(Block_tot))
            Order_ind = Rebuild_block_order(k, length(Block_tot))
            println("Order_ind = ",Order_ind') 
            #Uk.U = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
            #Uk.U_dag = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
            #println("Uk.U = ",Uk.U)
            #println("state_col = ",Uk.ind_state)
            #println("ind_U_col = ",Uk.ind_U_col)
            #x0 = CuArray(ones(Quantum_Model.n_sites,2))
            #x0[:,2] .= -1
            #ψ_x = ψ_G(rbm, Int.(x0),2)
            #println("+++++++++++++++++  ψx = ",ψ_x)
                # if k == nv  # 2 sites block
                #     bond_ind = [nv,1]
                # else
                #     bond_ind = [k,k+1]
                # end  
                
                #bond_ind = gen_bond(k,8) # 4 sites block device 1
                #bond_ind = collect(1:4) # all sites
                #n_bonds = length(bond_ind)
                #bond_b = Array{Int}(undef,0)
                #println("b == ", bond_b)
                # for k_b = 1 : n_bonds
                #     b_ind = bond_ind[k_b]
            
            #     b_n = collect((b_ind-1)*α_h + 1 : b_ind*α_h)
                #     bond_b = [bond_b;b_n]
                # end
                #bond_b = collect(1:nh)

                
                #println("bond =  ",bond_ind)
                #println("bond_b =  ",bond_b)
                #println("size of updated parameters = ", length(bond_ind)+length(bond_b) +length(bond_ind)*s_g.rbm_g._nh)


                #println("c = $c")

            #if Ham_counter == 1
            #	    rbm_t = RBM_Time_Initialization(rbm._nv, rbm._nh, rand(1:1000); sigma = 0.01)
            #else
            #	    rbm_t = RBM_Time(rbm.a[:], rbm.W[:,:])
            #end
            
            #if Ham_counter == 1
                #rbm_t = RBM_Time_Initialization(rbm._nv, rbm._nh , rand(1:1000); sigma = 0.01)
            #    rbm_t = RBM_Time(rbm.a[:], rbm.W[:,:])
                #rbm.W = [rbm.W[:,:]; CUDA.zeros(1 , rbm._nv)]
                #rbm._nh += 1
                #else
                #rbm_t = RBM_Time(rbm.a[:], rbm.W[:,:])
                #end
            rand_d = Normal(0, 0.1)
            rand_a_real = CuArray(rand(rand_d, rbm._nv))
            rand_a_imag = CuArray(rand(rand_d, rbm._nv))
            rand_W_real = CuArray(rand(rand_d, rbm._nh, rbm._nv))
            rand_W_imag = CuArray(rand(rand_d, rbm._nh, rbm._nv))
                #a_k = real.(rbm.a[:]) .* (1 .+ rand_a_real) .+ imag.(rbm.a[:]) .* (1 .+ rand_a_imag) *im
            #W_k = real.(rbm.W[:,:]) .* (1 .+ rand_W_real) .+ imag.(rbm.W[:,:]) .* (1 .+ rand_W_imag) *im
            #rbm_t = RBM_Time(a_k[:]*0.9, W_k[:,:]*0.9)
            c0 = 70
	        ct = sum(abs.(rbm.a[:])) + sum(abs.(rbm.b[:])) + sum(abs.(rbm.W[:,:]))
            #rbm_t = RBM_Time(rbm.a[:]*c0/ct*0.9, rbm.W[:,:]*c0/ct*0.9)
	        rbm_t = RBM_Time(rbm.a[:] , rbm.b[:], rbm.W[:,:])
            #rbm_t = RBM_Time_Initialization(rbm._nv, rbm._nh, rand(1:1000); sigma = 0.01)
                #rbm_t = RBM_Time(s_g.rbm_g.a, s_g.rbm_g.b, s_g.rbm_g.W)
                #rbm_t = RBM_Time(s_g.rbm_g.a*1.01, s_g.rbm_g.b*1.01, s_g.rbm_g.W*1.01)
                #a0,b0,W0 = Initial_abW(rbm._nv, s_g.rbm_g._nh ; sigma=0.01)
                
                #rbm_t.a[bond_ind]   .= rbm_t.a[bond_ind] 
                #rbm_t.b             .= rbm_t.b 
                #rbm_t.W[:,bond_ind] .= rbm_t.W[:,bond_ind] 
                #rbm_t = RBM_Time(a0,b0,W0)
            #println("s_g.state[:,1] =   ",s_g.state[:,1])
            #println("s_t.state_t[:,1] = ",s_t.state_t[:,1])
            #println("abW[] = ", rbm_t.W[3,3])
            Overlap_t = 0
            #println("Before updating: a b and W = ", rbm_t.a[1], "    ", rbm_t.W[1])
            #println("")
            rbm.γ = γ₀
            count_loop = 0
            overlap_min = 1
            while  Loop <= n_loop
                value_break = 0    
                if (Loop - 1)  % segment_redu_γ == 0 && Loop != 1 && rbm.γ > final_γ + 1e-9
                    rbm.γ = rbm.γ *LR_coff
                    println(">>>>Learning rate = ", rbm.γ)
                end
                println("Loop = ",Loop)
                #N_total =  s_g.n_sweeps * s_g.N_states
                #s_g_saved = CuArray(zeros(Int, Quantum_Model.n_sites, N_total))
                #s_t_saved = CuArray(zeros(Int, Quantum_Model.n_sites, N_total))
                #for iter_j = 1 : s_g.n_sweeps
                #	single_sweep_time!(Quantum_Model, s_g, rbm)
                #	single_sweep_time!(Quantum_Model, s_t, rbm_t, s_g)
                #	s_g_saved[: , (iter_j - 1) * s_g.N_states + 1 : iter_j * s_g.N_states] .= s_g.state[:,:]
                #	s_t_saved[: , (iter_j - 1) * s_g.N_states + 1 : iter_j * s_g.N_states] .= s_t.state_t[:,:]
                #end
                m_c = 0
                #for m = 1 : length(Block_ind)
                
                block_m = Block_ind[1]
                println("Block_m = ",block_m)
                for num_epoch_k = 1 : num_epochs
                    #bond_ind = [k_n,s_g.rbm_g.nv_bond[k_n]]
                    #println(" Bond = ", bond_ind)
                    #for j = 1 : num_epochs 
                    #bond_ind = [k,s_g.rbm_g.nv_bond[k]]
                
                    s_g.counter += 1
                    
                    Sum_NN_para = sum(abs.(rbm_t.a[:])) + sum(abs.(rbm_t.b[:])) + sum(abs.(rbm_t.W[:,:]))
                    #println("Sum_NN_para = ",Sum_NN_para)
                    io3 = open("sum_NN_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+")
                    println(io3, Loop, "     ", Sum_NN_para)
                    close(io3)
        


                    #s_t=Sampler_Time(nv, s_g.n_state)
                    #Overlap,  ∂θ  = run_time_SR_Trotter_Local(s_g, s_t, rbm_t, Uk, k, bond_ind, bond_b) 
                    #Overlap,  ∂θ  = run_time_SR_Trotter_Local_Unique(s_g, s_t, rbm_t, Uk, k, bond_ind, bond_b) 
                    #Overlap,  ∂θ  = run_time_SR_Trotter_Local_Unique_2_no_sweep(s_g, s_t, rbm_t, Uk, k, bond_ind, bond_b) 
                    if Sample_Mode == "Normal"  		   
                        Overlap,  ∂θ  = run_time_SR_Trotter(Quantum_Model, rbm, rbm_t, s_g, s_t, Uk, block_m, site_k)
                        #adam_counter[block_m] = adam_counter[block_m] .+ 1
                        #Overlap,  ∂θ  = run_time_SR_Trotter(Quantum_Model, rbm, rbm_t, s_g, s_t, Uk, block_m, site_k,s_g_saved, s_t_saved) 
                        update_parameters_block!(rbm_t, ∂θ, block_m, rbm.γ, rbm._nh)
                        #Adam_update!(adam_para, rbm_t, ∂θ, adam_counter, block_m, rbm._nh; β₁ = 0.9, β₂ = 0.999, α=0.001, ϵ = 1e-8)
                    elseif Sample_Mode == "IS"
                        Overlap = run_time_SR_Trotter_IS(Quantum_Model, rbm, rbm_t, s_g, s_t, Uk, block_m, site_k, tol)
                    end

                    #adam_counter[block_m] = adam_counter[block_m] .+ 1

                    #Overlap,  ∂θ  = run_time_SR_Trotter(s_g, s_t, rbm_t, Uk, k, "all")
                    Overlap_t = abs(Overlap)
                    overlap_min = min(Overlap_t, overlap_min)

                    io2 = open("Overlap_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+");
                    println(io2, Loop,"     ", Overlap)
                    close(io2)

                    #for j = 1 : length(block_m)
                    #println("j = $j", size(∂θ[j][:]))
    
                    #update_parameters_block!(rbm_t, ∂θ, block_m, rbm.γ, rbm._nh)
                    #end
    

                    #update_time_Local!(rbm_t, ∂θ, bond_ind, bond_b, s_g.rbm_g.α, s_g.rbm_g._nv, s_g.rbm_g._nh)
                    #println("======================")
                    #println("After updating: a b and W = ", rbm_t.a[1],"       ", rbm_t.W[1])
                    #println("sampler_g.rbm_g.a b and W = ", rbm.a[1],"       ",rbm.W[1])
                    #println("======================")
                    #update_time!(rbm_t, ∂θ,  s_g.rbm_g.α, s_g.rbm_g._nv, s_g.rbm_g._nh)
                    #if j == num_epochs
                    println("=====>>>  ",  "Iteration = ", s_g.counter, "   Overlap = ",Overlap)
        
                    #println("abW[] = ", rbm_t.W[3,3],"  ---------------------")
                    #end
                    if (abs((Overlap_t)) <= tol) && s_g.counter >= 10 #&& m_c == length(Block_ind)
                        value_break = 1	
                        println("Overlap_t = ",Overlap_t,"   Loop = ",Loop)
                        Print_txt(k, s_g.counter, nv , α_h, Total_sample, tol, random_seed_num, overlap_min)
                        break
                    end
                end  # for loop: num_epoch

                if value_break == 1
                    break
                end

                    

                #if (abs(Overlap_t) <= tol)
                    #    value_break = 1	
                    #    println("Overlap_t = ",Overlap_t,"   Loop = ",Loop)
                    #    Print_txt(k, s_g.counter, nv , α_h, Total_sample, tol, random_seed_num, Overlap_t)
                    #    break
                #end
                if Loop == n_loop
                    println("Overlap_t = ",Overlap_t,"   Loop = ",Loop )
                    Print_txt(k, s_g.counter,  nv, α_h, Total_sample, tol, random_seed_num, Overlap_t)
                end

                if value_break == 1
                    break
                end
                count_loop = Loop
                Loop += 1

            end   # for loop: loops
    
            # if count_loop == n_loop && rbm._nh <= rbm._nv * 20

            #     Random.seed!(rand(1:1000))
            #     d = Normal(0, 0.0001)
		    #     b_add = rand(d, rbm._nv) .+ im * rand(d, rbm._nv)
            #     W_add = rand( d, rbm._nv, rbm._nv) .+ im * rand(d, rbm._nv, rbm._nv)
            #     #rbm_t = RBM_Time_Initialization(rbm._nv, rbm._nh , rand(1:1000); sigma = 0.01) 
		    #     rbm_t = RBM_Time(rbm.a[:], [rbm.b[:]; b_add[:]], [rbm.W[:,:];W_add[:,:]])
            #     rbm.W = [rbm.W[:,:]; CUDA.zeros(rbm._nv , rbm._nv)]
            #     rbm._nh += rbm._nv
            #     continue
            # else
            #     count_k += 1
            # end


            rbm.a .= rbm_t.a
            rbm.b .= rbm_t.b
            rbm.W .= rbm_t.W
        
            s_g.state .= s_t.state_t
            # P_state, σ_x = Computing_Probabilities(s_g)
            # io = open("res time evolution $nv tol_$tol  Sampling_$total_s.txt", "a+");
            # println(io, "Probablities:")
            # println(io, P_state)
            # close(io)

        end  # for loop: Hamiltonian unitrary operator       

        # P_state, σ_x = Computing_Probabilities(s_g)
        # println("Time = ",evo.time[t_ind], "   Measurement= ", σ_x)
        # println("++++++++++++++++++++++++++++++++++++++++++++++")
        # Print_txt_time(evo.time[t_ind], Overlap_t, σ_x, nv, tol, total_s )
        
        σx = 0

        if mod(t_ind - 1, 1) ==  0 || t_ind == 2
            # for j = 1:5
            #     σx +=  Measurement_time(Quantum_Model, rbm, s_g, 50)
            # end
            σx = measurement_exact_sampling(Quantum_Model, rbm, s_g, 1)
            println("Time = ",evo.time[t_ind], "   Measurement= ", σx)
            CUDA.@allowscalar save("./Time_Quench_RBM_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(Quantum_Model.hx)_hz_$(Quantum_Model.hz[1]).jld2", "at", Array(rbm.a[:]), "bt", Array(rbm.b[:]), "Wt",Array(rbm.W[:,:]))
        

            #Print_txttxt_measure(evo.time[t_ind], Overlap_t, σx/5, nv, tol, total_s )
            Print_txt_measure(nv, α_h, Total_sample, tol, random_seed_num, evo.time[t_ind], σx)
        end

        println("++++++++++++++++++++++++++++++++++++++++++++++")
        #Print_txttxt_measure(evo.time[t_ind], Overlap_t, σx/5, nv, tol, total_s )
        #Print_txt_measure(nv, α_h, Total_sample, tol, random_seed_num, evo.time[t_ind], σx/5)
    end # for loop: time point
end

# Runge-Kutta method
function main_rungle_kutta(quant_sys::T1, rbm::RBM, sample::T4, rk::rungle_kutta, evo::Time_Evolution,  file_name::String)where {T1<:Abstract_Quantum_System, T4 <: Abstract_Sampling}
    println("===> update all parameters !")
    block_m = collect(1 : rbm._nv)
    n_params = rbm._nv + rbm._nh + rbm._nv * rbm._nh 
    println("number of parameters is: $n_params")
    tn = length(evo.time)
    rbm_t0 = RBM_Time(rbm.a[:] , rbm.b[:], rbm.W[:,:])
    for  t_ind = 2 : length(evo.time)
        #println("Time = ",evo.time[t_ind])
        
        rbm_t0.a .= copy.(rbm.a)
        rbm_t0.b .= copy.(rbm.b)
        rbm_t0.W .= copy.(rbm.W)

        # thermalization
        for j = 1 : sample.n_thermals
            single_sweep_time!(quant_sys, sample, rbm)
        end


        # nn_t0.b .= copy.(nn.b)
        # nn_t0.W .= copy.(nn.W)
        kn_save = Array{CuArray{ComplexF64}}(undef,rk.order)
        for n = 1 : rk.order
            overlap, kn = optimize_params(quant_sys, rbm, sample, n_params, rk)
            println("E_avg = $overlap")
            if abs(overlap) >= 1e+8 || isnan(overlap)
                error("Wrong ground energy: $overlap !")
            end
            kn_save[n] = kn
            if n != 3
                update_parameters_rk!(rk, rbm, rbm_t0, 0.5 * evo.dt * kn)
            elseif n == 3
                update_parameters_rk!(rk, rbm, rbm_t0, evo.dt * kn)
            end
        end
        ∂θ = sum(rk.k .* kn_save) / 6 * evo.dt
        update_parameters_rk!(rk, rbm, rbm_t0, ∂θ)
        

        if mod(t_ind - 1, 10) ==  0 || t_ind == 2
            if quant_sys.n_sites <= 14
                σx = σx = measurement_exact_sampling(quant_sys, rbm, sample, 1)
            else
                σx = Measurement_Metropolis_Sampling(quant_sys, nn.model,  sample, σ_i)
            end
            io_time = open(file_name,"a+")
            println(io_time, evo.time[t_ind],": ", "  σx = ", σx)
            close(io_time)
            println("Time = ",evo.time[t_ind], "   Measurement= ", σx)
            #save("./Time_Quench_FNN_rk$(rk.order)_$(quant_sys.n_sites)_sites_$(sample.n_states)_tol_$(tol)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(quant_sys.hx)_hz_$(quant_sys.hz[1]).jld2", "fnn", nn, "act_func", typeof(act_func))
        end
        total_samples = sample.N_states * n_sweeps
        hz1 = CUDA.@allowscalar quant_sys.hz[1]
        if mod(t_ind - 1, 10) ==  0 || t_ind == 2
            CUDA.@allowscalar save("./Time_Quench_RBM_rk_$(rbm._nv)_sites_$(rbm._nh)_$(total_samples)_time_$(evo.time[t_ind])_dt_$(evo.dt)_hx_$(quant_sys.hx)_hz_$(quant_sys.hz[1]).jld2", "at", Array(rbm.a[:]), "bt", Array(rbm.b[:]), "Wt",Array(rbm.W[:,:]))
        end
    end # end for loop: time steps

end


function update_parameters_block!(rbm_t :: RBM_Time{T}, ∂θ::CuArray{T,1}, nm::CuArray{Int,1}, γ :: Float64, nh:: Int) where T # update a and W
    rbm_t.a[nm] .-= γ * ∂θ[1:length(nm)] 
    rbm_t.b .-= γ * ∂θ[length(nm) + 1 : length(nm) + nh]
    rbm_t.W[:, nm] .-= γ * reshape(∂θ[length(nm) + nh + 1 : end], (nh, length(nm)))
end

function update_parameters_block_IS!(rbm_n :: RBM_n{T}, rbm:: RBM, ∂θ::CuArray{T,1}, nm::CuArray{Int,1}) where T
	rbm_n.a[nm] .-= rbm.γ * ∂θ[1:length(nm)]
    rbm_n.b .-= γ * ∂θ[length(nm) + 1 : length(nm) + nh]
	rbm_n.W[:, nm] .-= rbm.γ * reshape(∂θ[length(nm) + nh + 1 : end], (rbm._nh, length(nm)))
end

function update_parameters_rk!(rk::rungle_kutta, rbm::RBM, rbm_t0::RBM_Time,∂θ::CuArray{T,1}) where T # update a and W
    nm = collect(1 : rbm._nv)
    nh = rbm._nh
    rbm.a .= rbm_t0.a .+ ∂θ[1:length(nm)] 
    rbm.b .= rbm_t0.b .+ ∂θ[length(nm) + 1 : length(nm) + nh]
    rbm.W .= rbm_t0.W .+ reshape(∂θ[length(nm) + nh + 1 : end], (nh, length(nm)))
end


function Print_txt(k::T, Loop::T, nv::T, α_h::T, Total_sample::T, tol::Float64, random_seed_num::T,  Overlap::Number) where{T<:Int}
    io = open("Time_Evolution_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+");
    println(io, k," --->  ", Loop,"     ", Overlap)
    close(io)
end

function Print_txt_measure(nv::T, α_h::T, Total_sample::T, tol::Float64, random_seed_num::T, evo_time::Float64, σx ::ComplexF64)where{T<:Int}
    io = open("Time_Evolution_$(nv)_sites_$(α_h)_$(Total_sample)_tol_$(tol)_$(random_seed_num).txt", "a+");
    println(io, evo_time,": ", "  σx = ", σx)
    close(io)
end

function measurement_exact_sampling(ising::Ising_Model, rbm::RBM, s_g::Sampling, site_n::Int) 
    
    σ_avg = 0 
    σ_avg = σ_local_exact_sampling(ising.n_sites, rbm, site_n)
    #println("Exact Measurment") 
    return σ_avg
end

function σ_local_exact_sampling(L::Int, rbm::RBM,  i) 
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
    exact_ψ = ψ_G(rbm, exact_state, N_Exact)
    #exact_lnψ = model(exact_state) 
    #exact_lnψ_ratio = CUDA.@allowscalar exp.(exact_lnψ .- exact_lnψ[1])
    exact_ψ_ratio = CUDA.@allowscalar (exact_ψ ./ exact_ψ[1])
    P_ψ = abs.(exact_ψ_ratio).^2 / sum(abs.(exact_ψ_ratio).^2)
    #println("P = $P_ψ")
    X = σ_flip(exact_state[:,:] , i)

    #σ_loc = 0.0+0.0im
    new_states = X
    #println("new_state = $new_state")
    #σ_loc = Ratio_ψ(rbm, s_g.state, flip_global)
    σ_loc = ψ_G(rbm, new_states, N_Exact)./ψ_G(rbm, exact_state[:,:], N_Exact)
    #σ_loc = ratio_ψ(model, new_states, exact_state[:,:])
    #println(size(σ_loc))
    return sum(σ_loc .* P_ψ)
end


function Measurement_time(ising::Ising_Model, rbm::RBM, s_g::Sampling, N_total :: Int)
    #println("Sampler_a = ",st.rbm.a)
    #thermalize_G!(s_g)
    σ_avg = 0 
    #Num = Int(round(N_total/N))
    #println("Num = ",Num)
    for j =1 : N_total 
        for k = 1:s_g.n_sweeps
            single_sweep_time!(ising, s_g, rbm)
        end
        σ_avg += σ_local(ising.n_sites, rbm, s_g, 1)
    end
    #println(n*N)
    return σ_avg/(N_total * s_g.N_states)
end

function σ_local(L::Int, rbm::RBM{T}, s_g::Sampling, i) where{T<:ComplexF64}
    #n = size(initial_state,1)
    X = σ_flip(s_g.state[:,:],i)
    N = s_g.N_states

    #σ_loc = 0.0+0.0im
    new_state = X
    flip_local = CuArray(zeros(Int,N) .+ i) 

    flip_global = gen_flip_list(flip_local, L)
    #σ_loc = Ratio_ψ(rbm, s_g.state, flip_global)
    σ_loc = ψ_G(rbm, new_state, N)./ψ_G(rbm, s_g.state[:,:], N)
    #println(size(σ_loc))
    return sum(σ_loc)
end

function σ_flip(x::CuArray{Int},i)
    X_new = copy(x);
    X_new[i,:] .= -X_new[i,:];
    return X_new;
end

#=======================#
#=======================#

########### --- Main Code: Initialization --- ###########
########### --- Main Code: Initialization --- ###########

export Neural_Net_Initialization_parallel
function Neural_Net_Initialization_parallel_Time(::Type{T}, quan_sys_setting::Tuple, neural_net_setting::Tuple, sample_setting::Tuple, update_setting::Tuple, NN_parameters:: Tuple, Time_setting::Tuple, tol::Float64, N_Ham_bond ::Int) where{T<:Number}
    
    # basic setting
    quan_sys_name = quan_sys_setting[1]
    Lattice = quan_sys_setting[2]
    Lx = Lattice[1]
    Ly = Lattice[2]
    n_sites = Lx*Ly

    n_thermals = sample_setting[1]
    n_sweeps = sample_setting[2]
    N = sample_setting[3] #How many states updated in one step

    # set quantum model
    if quan_sys_name == "Ising"

        J = quan_sys_setting[3]
        hx = quan_sys_setting[4]
        hz = quan_sys_setting[5]
        boundary_condition = quan_sys_setting[6]
	    correlation_num = quan_sys_setting[7]
        Quantum_Model = Ising_Model_Initialization(J, hx, hz, Lx, Ly, boundary_condition)
        ini_states = rand(-1:2:1, n_sites, N)
        sample = Sampling(n_thermals, n_sweeps, N, 0, ini_states)
	    Sample_Mode = sample_setting[4]
    elseif quan_sys_name == "Heisenberg"
        J = quan_sys_setting[3]
        boundary_condition = quan_sys_setting[4]
        Quantum_Model = Heisenberg_Model_Initialization(J, Lx, Ly, boundary_condition)
        ini_states = rand(-1:2:1, n_sites, N)
        sample = Sampling(n_thermals, n_sweeps, N, 0, ini_states)
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
    if Sample_Mode != "Normal" && Sample_Mode != "IS"
	    error("No such option!   The only options are: Normal and IS.")
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

    load_NN_p = NN_parameters[1]
    
    if load_NN_p == 1
	    Neural_Net.a = deepcopy(NN_parameters[2])
	    Neural_Net.b = deepcopy(NN_parameters[3])
	    Neural_Net.W = deepcopy(NN_parameters[4])
	    #println("The initial parameters are loaded:  a[1] = $(Neural_Net.a[1]),  b[1] = $(Neural_Net.b[1]),   W[1] = $(Neural_Net.W[1])")
    else
	    error("1 means parameters are needed to load")
    end

    t_0 = Time_setting[1]
    dt = Time_setting[2]
    t_f = Time_setting[3]

    evo = Time_Evolution(t_0, t_f, dt)





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

    if iteration_mode[1] == "tvmc"
        order = iteration_mode[2]
        rk_k_coff = iteration_mode[3]
        rk = rungle_kutta(order, rk_k_coff)
    end

    # if iteration_mode != "SR" && iteration_mode != 0
	#     error("There is no such iteration mode. /the option are: SR and 0 wihch means normal method without SR.")
    # end

    # record settings
    output_file_name = "Time_Evolution_$(n_sites)_sites_$(neural_net_setting[2])_$(N * n_sweeps)_tol_$(tol)_$(random_seed_num).txt"
    io = open("Time_Evolution_$(n_sites)_sites_$(neural_net_setting[2])_$(N * n_sweeps)_tol_$(tol)_$(random_seed_num).txt", "w");
    io = open("Time_Evolution_$(n_sites)_sites_$(neural_net_setting[2])_$(N * n_sweeps)_tol_$(tol)_$(random_seed_num).txt", "a+");
    println(io, "Date:  ",(today()),"  ", Dates.Time(Dates.now()))
    println(io, "The ground energy of $quan_sys_name model, and the total site is $n_sites, where Lx = $Lx, Ly = $Ly. ")
    println(io, "The boundary condition is: $boundary_condition.  Sample Mode is $Sample_Mode") 
    if quan_sys_name == "Ising"
	    println(io,"J = $J,   hx = $hx")
	    println(io, "hz = $hz \n")
    end
    if neural_net_name == "RBM"
        println(io, "for RBM, nv = $n_sites, nh = $nh. Learning rate γ = $α, The cofficient of reducing learning rate redu_coff_γ = $redu_coff_α, final learning rate = $final_γ, segment_redu_γ = $segment_redu_γ")
    end
    
    println(io, "The mode of optimization is $update_mode, and the block size_x = $block_size, the block size on y direction = $(block_size_xy[2]) the shift amount = $n_shift_site, iteration mode is $(iteration_mode[1]).")
    println(io,"The max number of loops for each Hamiltonian is $n_loop")
    println(io,"The tolerance is $tol")
    println(io, "For the sampling, the number state updated in one step is: $N, the number of thermailization is: $n_thermals, the number of sweeps is $n_sweeps.")
    println(io, "Thus, the total state is $(N * n_sweeps). \n")
    println(io,"The random seed number is:  $random_seed_num")
    CUDA.@allowscalar println(io, "a[1] = ", Neural_Net.a[1], "  W[1] = ", Neural_Net.W[1])

    σx = 0
    σx = measurement_exact_sampling(Quantum_Model, Neural_Net, sample, 1)
    # for j = 1:5
	#     σx +=  Measurement_time(Quantum_Model, Neural_Net, sample, 50)
    # end
    println(io, "At the begining, the value of σx = ", σx)

    println(io, "Initial Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)
    
    #io2 = open("Correlation_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(N * n_sweeps)_$(random_seed_num).txt", "w");
    #io2 = open("Correlation_$(n_sites)_sites_$(neural_net_setting[2])_$(block_size)_$(N * n_sweeps)_$(random_seed_num).txt", "a+");
    #println(io2, "Correlation Number =  $correlation_num  ")
    #close(io2)

    #Optimize_gs_parallel(Quantum_Model, Lx, Ly, Neural_Net, sample, block_size_xy, num_epochs, n_loop, n_shift_site, update_mode, redu_coff_α, final_γ, segment_redu_γ,  iteration_mode, random_seed_num, correlation_num, n_process)
    if iteration_mode[1] == "tvmc"
	    main_rungle_kutta(Quantum_Model, Neural_Net, sample, rk, evo, output_file_name)
    else
        optimize_t_Trotter_time_Local_block(Quantum_Model, Lx, Ly, Neural_Net, block_size_xy, num_epochs, n_loop, sample, evo, tol, update_mode, redu_coff_α, final_γ,  segment_redu_γ, random_seed_num, N_Ham_bond, Sample_Mode)
    end
    io = open("Time_Evolution_$(n_sites)_sites_$(neural_net_setting[2])_$(N * n_sweeps)_tol_$(tol)_$(random_seed_num).txt", "a+");
    println(io, "Finish Time:  ",(today()),"  ", Dates.Time(Dates.now()))
    close(io)


   # println("a = ", typeof(Neural_Net.a))

end