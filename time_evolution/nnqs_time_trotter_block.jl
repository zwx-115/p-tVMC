# this code is for generating trotter blocks:

########### ---  Time Evolution Type  --- ###########
########### ---  Time Evolution Type  --- ###########

abstract type abstract_evolution end # Define type of evolution: trotter decomposition , taylor, RK2 or RK4, ...

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
