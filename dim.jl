#LOAD NECESSARY LIBRARIES

using Random, Plots
using Zygote, ForwardDiff
using OrdinaryDiffEq, DiffEqSensitivity
using DiffEqCallbacks
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles
using YAML
using CSV
using DataFrames



# SETUP TRAINING CONDITIONS 
lb = 1.e-8
ENV["GKSwstype"] = "100"
n_epoch = 100
maxiters= 500
adam_lr= 1.e-3
lr_max= 2.5e-5
lr_min= 5.e-5
lr_decay= 0.2
lr_decay_step= 500
w_decay= 1.e-7 
grad_max= 1.e2


#LOAD FILES 
file_path = "c:/Users/Arjun/Desktop/CRNN/Anode_electrolyte/exp_data/dataset.csv"
data = CSV.read(file_path, DataFrame)
time_exp = data[:, 1]  # Column 1 is time in min
temp_exp = data[:,2] #Column 2 is temp of sample in °C
heat_flow = data[:, 3]    # Column 3 is heat flow

plot(temp_exp, heat_flow, xlabel="Temperature", ylabel="Heat Flow", title="Temperature vs. Heat Flow", lw=2, legend=false)


# TRAINABLE KINETIC PARAMETERS DEFINITION + INITIALIZATION
np = 21+1 #with slope at the end

p = randn(Float64, np) .* 1.e-2;

p[1:4] .+= 1;  # A
Ea_IC=[1.0, 1.1, 1.2, 1.3] #Ea initial condition. Roughly initialized.
p[5]+=Ea_IC[1] 
p[6]+=Ea_IC[2] 
p[7]+=Ea_IC[3] 
p[8]+=Ea_IC[4] 

p[13] += 0.2; #delta H, roughly guessed based on how big the peaks in the data sort of look
p[14] += 1; #delta H
p[15] += 0.8; #delta H
p[16] += 0.5; #delta H
p[17:20] .+= 1; #Reaction orders
p[21]= 0.5; #Alpha conversion factor
p[22]= 0.1; #slope, as per original CRNN code


# SOME FUNCTIONS
function p2vec(p)
    #some clamps in place during debugging to make sure none of the parameters get too large or small 
    #these don't appear to be necessary in the final runs

    slope = p[end] .* 1.e1
    w_A = p[1:4] .* (slope * 20.0) #logA
    w_A = clamp.(w_A, 0, 50) 

    
    w_in_order=p[17:20] #rxn orders
    w_in_order=clamp.(w_in_order, 0.01, 10)

    w_in_Ea = abs.(p[5:8]) #Ea
    w_in_Ea = clamp.(w_in_Ea, 0.0, 3)

    w_in_b = (p[9:12]) #non-exponential temp dependence, can be negative, no clamp

    w_delH = abs.(p[13:16])*100
    w_delH=clamp.(w_delH, 10, 300) 

    w_alpha = p[21]
    w_alpha = clamp.(w_alpha,0,1)

    return w_in_Ea, w_in_b, w_delH, w_in_order, w_A, w_alpha
end

function getsampletemp(t, T0, beta)
    if beta[1] < 100
        T = T0 .+ beta[1] / 60 * t  # K/min to K/s
    end
    return T
end

#CRNN function DEFINITION

const tsei1r = 1.0 #Thickness of SEI in mm at 100°C
const tsei2r = 1.0 #Thickness of SEI from final condition of peak 1 or from literature
const R = -1.0 / 8.314  # J/mol*K

function crnn!(du, u, p, t)
    # Extract parameters from p
    w_in_Ea, w_in_b, w_delH, w_in_order, w_A, w_alpha = p2vec(p)

    
    conc = [u[1], u[3], u[3], u[5]]
    thick = [u[2], u[4]]

    # Apply logarithmic transformation with safe clamping
    logX = @. log(clamp(conc, lb, 10.0))
    S = @. clamp(thick, lb, 1)

    # Ensure T is correctly calculated
    T = getsampletemp(t, T0, beta)
    temp_term = reshape(hcat(log(T), R / T) * hcat(w_in_b, w_in_Ea * 1e5)', 4)
    rxn_ord_term = w_in_order .* logX

    pre_rxn_rates = temp_term + rxn_ord_term + w_A
    pre_rxn_rates[1] -= S[1] / tsei1r
    pre_rxn_rates[2] -= S[2] / tsei2r

    # Exponentiate the computed rates
    pre_rxn_rates = @. exp(pre_rxn_rates)

    # Assign to du
    du[1] = -(pre_rxn_rates[1])
    du[2] = pre_rxn_rates[1]
    du[3] = -(pre_rxn_rates[2] + pre_rxn_rates[3])
    du[4] = -pre_rxn_rates[2]
    du[5] = -pre_rxn_rates[4]
end

function HRR_getter(times, u_outputs, p)
    # Extract parameters
    w_in_Ea, w_in_b, w_delH, w_in_order, w_A, w_alpha = p2vec(p)

    num_times = length(times)

    # Confirm dimensions
    println("In HRR_getter:")
    println("Number of time points (num_times): ", num_times)
    println("Size of u_outputs: ", size(u_outputs))

    # Initialize arrays based on the element type in u_outputs
    conc_prof = zeros(eltype(u_outputs[1,1]), num_times, 4)
    thick_prof = zeros(eltype(u_outputs[1,1]), num_times, 2)
    println("Size of conc_prof: ", size(conc_prof))
    println("Size of thick_prof: ", size(thick_prof))

    # Assign values without type mismatch
    conc_prof[:, 1] .= u_outputs[1, :]
    conc_prof[:, 2] .= u_outputs[3, :]
    conc_prof[:, 3] .= u_outputs[3, :]
    conc_prof[:, 4] .= u_outputs[5, :]

    thick_prof[:, 1] .= u_outputs[2, :]
    thick_prof[:, 2] .= u_outputs[4, :]

    # Print after assignment
    println("After assignment:")
    println("Size of conc_prof: ", size(conc_prof))
    println("Size of thick_prof: ", size(thick_prof))

    # Ensure kinetic parameters are repeated for each time point
    println("Size of w_in_b: ", size(w_in_b))
    replicated_w_in_b = repeat(w_in_b', num_times, 1)
    println("Size of replicated_w_in_b: ", size(replicated_w_in_b))

    println("Size of w_in_Ea: ", size(w_in_Ea))
    replicated_w_in_Ea = repeat((w_in_Ea' * 1e5), num_times, 1)
    println("Size of replicated_w_in_Ea: ", size(replicated_w_in_Ea))

    println("Size of w_A: ", size(w_A))
    replicated_w_A = repeat(w_A', num_times, 1)
    println("Size of replicated_w_A: ", size(replicated_w_A))

    println("Size of w_in_order: ", size(w_in_order))
    replicated_w_in_order = repeat(w_in_order', num_times, 1)
    println("Size of replicated_w_in_order: ", size(replicated_w_in_order))

    log_conc_prof = @. log(clamp(conc_prof, lb, 10.0))
    println("Size of log_conc_prof: ", size(log_conc_prof))

    S_prof = @. clamp(thick_prof, lb, 1)
    println("Size of S_prof: ", size(S_prof))

    # Temperature-dependent terms
    T = getsampletemp(times, T0, beta)
    println("Size of T: ", size(T))

    # Reshape T for broadcasting
    log_T = log.(T)
    println("Size of log_T: ", size(log_T))

    temp_term1 = log_T .* replicated_w_in_b
    println("Size of temp_term1: ", size(temp_term1))

    temp_term2 = (R ./ T) .* replicated_w_in_Ea
    println("Size of temp_term2: ", size(temp_term2))

    temp_term = temp_term1 .+ temp_term2
    println("Size of temp_term: ", size(temp_term))

    rxn_ord_term = replicated_w_in_order .* log_conc_prof
    println("Size of rxn_ord_term: ", size(rxn_ord_term))

    # Combining terms and calculating reaction rates
    pre_rxn_rates = temp_term + rxn_ord_term + replicated_w_A
    println("Size of pre_rxn_rates: ", size(pre_rxn_rates))

    # Adjustments for specific reactions
    pre_rxn_rates[:, 1] .= pre_rxn_rates[:, 1] .- S_prof[:, 1] / tsei1r
    pre_rxn_rates[:, 2] .= pre_rxn_rates[:, 2] .- S_prof[:, 2] / tsei2r

    pre_rxn_rates = clamp.(pre_rxn_rates, -700, 700)

    rxn_rates = @. exp(pre_rxn_rates)
    println("Size of rxn_rates: ", size(rxn_rates))

    # Compute heat release
    #heat_rel = rxn_rates .* w_delH
    #println("Size of heat_rel: ", size(heat_rel))

    return rxn_rates
end


tspan = [0.0, 1.0];
u0 = [1.0,1.0,1.0,1.0,1.0];
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb)

condition(u, t, integrator) = u[1] < lb * 10.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

# Use an ODE algorithm without autodiff
alg = AutoTsit5(TRBDF2(autodiff = true));
sense = ForwardSensitivity(autojacvec = true)

function pred_n_ode(p)
    global beta = 5  # 5K/min
    global T0 = 100 + 273.15  # degrees K

    # Extract parameters
    w_in_Ea, w_in_b, w_delH, w_in_order, w_A, w_alpha = p2vec(p)

    ts = time_exp
    tspan = (ts[1], ts[end])

    # Initial conditions
    u0 = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Define the ODE problem with the current parameters
    prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb)

    # Solve the species trajectory
    sol = solve(
        prob,
        alg,
        saveat = ts,
        sensealg = sense,
        maxiters = maxiters,
    )

    # Convert `sol.u` to a 5 x num_times matrix to avoid nesting
    u_outputs = hcat(sol.u...)  # Each column is one time step

    # Post-processing: compute the heat release
    heat_rel = HRR_getter(ts, u_outputs, p) * w_delH  # Broadcasting .* for element-wise operation


    if sol.retcode == :Success
        nothing
    else
        @warn "Solver failed with beta: $(beta)"
    end

    return heat_rel, ts, sol
end


function loss_neuralode(p)
    pred = pred_n_ode(p)[1]
    loss = mae(pred, heat_flow)  # For heat releases
    return loss
end



#TRAINING



opt = ADAMW(adam_lr, (0.9, 0.999), w_decay);

# Assume `data` is your single experiment dataset
epochs = ProgressBar(1:n_epoch)
loss_epoch = zeros(Float64, n_epoch) # Track loss over each epoch
grad_norm = zeros(Float64, n_epoch)  # Track gradient norm over each epoch

for epoch in epochs  # loop through epochs
    global p
    # Calculate gradient with ForwardDiff on the single dataset
    grad = ForwardDiff.gradient(loss_neuralode, p)
    grad_norm[epoch] = norm(grad, 2)

    # Apply gradient clipping if necessary
    if grad_norm[epoch] > grad_max
        grad = grad ./ grad_norm[epoch] .* grad_max
    end

    # Update parameters
    update!(opt, p, grad)

    # Calculate and store the loss for this epoch
    loss_epoch[epoch] = loss_neuralode(p)
    loss_train = mean(loss_epoch[1:epoch])  # Mean loss up to current epoch

    # Update progress bar with loss and gradient norm
    set_description(
        epochs,
        string(
            @sprintf(
                "Epoch %d | Loss: %.2e | Grad Norm: %.2e",
                epoch,
                loss_train,
                grad_norm[epoch],
            )
        ),
    )

    # Optional: Run a callback for logging/plotting if desired
    #cb(p, loss_train, grad_norm[epoch])  # plotting or logging function
end

# If you have a callback for a final update (e.g., visualization)
#cbi(p)
