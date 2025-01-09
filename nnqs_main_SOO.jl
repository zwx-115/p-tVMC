# this code is for SOO method


function main_train_time_trotter_SOO(::Type{T}, quant_sys::T1, nn::FNN, nn_t::FNN_time, sample::T4,
    update_type::abstract_update, opt_type,
    n_epochs::Int, n_loops::Int, n_ham_bond::Int,
    evo::time_evolution, tol::Float64, σ_i::Int, seed_num::Int,
    block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
    file_name::String) #=
    =# where {T<:Number, T1<:abstract_quantum_system, T3 <: abstract_activation_function, T4 <: abstract_sampling}
    
    ### generate list of blocks of input and output
    ### +++++++++++++++++++++++++++++++++++++++++++
    block1_in_list = [];
    block1_out_list = [];
    block2_in_list = [];
    block2_out_list = [];
    block3_in_list = [];
    block3_out_list = [];
    for _ in 1:4
        push!(block1_in_list, collect(1:40))
        push!(block3_out_list, [1])
    end
    # block in:

    push!(block2_in_list, collect(1:80), collect(41:120), collect(81:160),collect(41:120))
    push!(block3_in_list, collect(1:60), collect(31:90), collect(61:120),collect(31:90))

    # block out:
    push!(block1_out_list, collect(1:80), collect(41:120), collect(81:160),collect(41:120))
    push!(block2_out_list, collect(1:60), collect(31:90), collect(61:120),collect(31:90))
    for k in  1 : 4
        println("the block $(k) ===>")
        println(" l1 in=> $(block1_in_list[k][1])<-->$(block1_in_list[k][end]), l1 out=> $(block1_out_list[k][1])<-->$(block1_out_list[k][end]) \n l2 in=> $(block2_in_list[k][1])<-->$(block2_in_list[k][end]), l2 out=> $(block2_out_list[k][1])<-->$(block2_out_list[k][end]) \n l3 in=> $(block3_in_list[k][1])<-->$(block3_in_list[k][end]), l3 out=> $(block3_out_list[k][1])<-->$(block3_out_list[k][end])")
        println("----------")
    end
    
    sleep(10)
    # # if upate all parameters
    # if block_in_size == "all" && block_out_size == "all" && layer_type == "all"
    #     layer_list = ["all"]
    #     block_in_list = [["all"]]
    #     block_out_list = [["all"]]
    # elseif layer_type == "layer"
    #     # We will use SOO optimization to update parameters:
    #     layer_list = [1, 2]
    #     #layer_list = [layer_list ; reverse(layer_list[2:end-1])]
    #     #layer_list = [1 , 2]
    #     # if update parameters in each block
    #     if block_in_size == "all" 
    #         for k = 2 : 2 #nn.n_layers - 1
    #             block_in_k = generate_blocks(nn.layer_size[k], nn.layer_size[k], 0)
    #             push!(block_in_list, block_in_k)
    #         end
    #     else 
    #         for k = 2 : 2 #nn.n_layers - 1
    #             block_in_k = generate_blocks(nn.layer_size[k], block_in_size[k], overlap_in[k])
    #             push!(block_in_list, block_in_k)
    #         end
    #     end
        
    #     if block_out_size == "all" 
    #         for k = 3 : 3 #nn.n_layers 
    #             block_out_k = generate_blocks(nn.layer_size[k], nn.layer_size[k], 0)
    #             push!(block_out_list, block_out_k)
    #         end
    #     else 
    #         for k = 3 : 3 #nn.n_layers 
    #             block_out_k = generate_blocks(nn.layer_size[k], block_out_size[k-1], overlap_out[k-1])
    #             push!(block_out_list, block_out_k)
    #         end
    #     end
    # end

    
    #block_out_list[1] = [block_out_list[1]; reverse(block_out_list[1][2:end-1])]
    
    
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
            dt = Normal(0, 0.05)
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

                for k in 1 : 4 # for loop of blocks
                    
                    println("number of iterations = $(sample.counter)")
                    if typeof(update_type) == gradient_descent
                        if (sample.counter - 1)  % update_type.decay_interval == 0 && sample.counter != 1 && update_type.γ> update_type.final_γ + 1e-9
                            update_type.γ = update_type.γ * update_type.decay_rate
                            println(">>>>Learning rate = ", update_type.γ)
                        end
                    end
                    println("$(k): ($(block1_in_list[k][1])<->$(block1_in_list[k][end]); $(block1_out_list[k][1])<->$(block1_out_list[k][end])),($(block2_in_list[k][1])<->$(block2_in_list[k][end]); $(block2_out_list[k][1])<->$(block2_out_list[k][end])),($(block3_in_list[k][1])<->$(block3_in_list[k][end]); $(block3_out_list[k][1])<->$(block3_out_list[k][end]))")
                    nb1 = length(block1_out_list[k]) * length(block1_in_list[k]) + length(block1_out_list[k])
                    nb2 = length(block2_out_list[k]) * length(block2_in_list[k]) + length(block2_out_list[k])
                    nb3 = length(block3_out_list[k]) * length(block3_in_list[k]) + length(block3_out_list[k])
                    println("nb1 = ",nb1)
                    println("nb2 = ",nb2)
                    println("nb3 = ",nb3)
                    n_params = nb1 + nb2 + nb3
                    println("number of parameters = ", n_params)

                    for epoch = 1 : n_epochs
                        sleep(5)
                        # if mod(sample.counter, 1) == 0 && typeof(sample) == Sampling
                        #     single_sweep!(quant_sys, nn, sample)
                        #     single_sweep_time!(quant_sys, nn_t, sample_t)
                        # end
                        overlap, ∂θ = optimize_params3(T, quant_sys, nn, nn_t, 
                        sample, sample_t, n_params, Um, site_m, 
                        block1_in_list[k], block1_out_list[k], block2_in_list[k], block2_out_list[k], block3_in_list[k], opt_type)
                        if abs(overlap) >= 1e+8 || isnan(overlap)
                            error("Wrong ground energy: $overlap !")
                        end
                        update_params3!(update_type, nn_t, ∂θ[:], block1_in_list[k], block1_out_list[k], block2_in_list[k], block2_out_list[k], block3_in_list[k])
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
                            println(io_time, m, " --->", sample.counter, "      $overlap_min", "      $overlap_t")
                            close(io_time)
                            break
                        end
                        
                        #println("t = ", update_type.t)

                    end # for loop: epoch

                    if value_break == 1
                        break
                    end

                    if value_break == 1
                        break
                    end
                end # for loop: block

                if loop == n_loops
                    println("overlap_t = ",overlap_t,"   loop = ",loop)
                    io_time = open(file_name,"a+")
                    println(io_time, m, " --->", sample.counter, "      $overlap_min", "      $overlap_t")
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
        else
            n_total_save = 2^(quant_sys.n_sites)
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


function neural_network_initialization_time_SOO(quant_sys_setting::Tuple, nn_setting::Tuple, 
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
    elseif opt_name == "SREnergy"
        λ = opt_setting[2]
        α_r = opt_setting[3]
        E_val = opt_setting[4]
        E_c = opt_setting[5]
        opt_type = sr_energy(λ, α_r, E_val, E_c)
    elseif opt_name == "CV"
        λ = opt_setting[2]
        α_r = opt_setting[3]
        c = opt_setting[4]
        opt_type = cv_optimization(λ, α_r, c)
    else
        error("There is no such a optimization method: $opt_name. The options of optimization method are: SR and tvmc.")
    end
    
    output_file_name = "time_evolution_SOO_$(nn_name)_$(n_sites)_sites_$(n_total)_$(sample_name)_tol_$(tol)_$(random_seed_num).txt"
    
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
    println(io,"the type of evolution is: $evo_mode, the number of bond is $n_ham_bond")
    

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

    
    
    main_train_time_trotter_SOO(T, quant_sys, neural_net, neural_net_time, 
                        sample, update_type, opt_type, n_epochs, n_loops, n_ham_bond,
                        evo, tol, measurment_site, random_seed_num,
                        block_in_size, block_out_size, overlap_in, overlap_out, layer_type,
                        output_file_name)
    
    io = open(output_file_name, "a+")
    println(io, "Final Time:  ",(today()),"  ", Dates.Time(Dates.now()))  
    close(io)
end  