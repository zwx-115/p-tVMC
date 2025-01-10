# This code is for the measurement:

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
