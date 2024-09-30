using Plots
using DifferentialEquations
using Lux, DiffEqFlux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random
using ComponentArrays
using OrdinaryDiffEq

# STEP 1: GENERATE SYNTHETIC DATA

# Parameters
beta = 0.3  # Transmission_rate
gamma = 0.1  # Recovery_rate
N = 1000     # Total number of people

function SIR!(du, u, p, t)
    du[1] = -beta * u[1] * u[2] / N
    du[2] = beta * u[1] * u[2] / N - (gamma * u[2])
    du[3] = gamma * u[2]
end

u0 = [999, 1, 0]
tspan = (0, 160)

prob = ODEProblem(SIR!, u0, tspan)
sol = solve(prob, Tsit5(); saveat = 2)

t = sol.t  # Time_points
S = [u[1] for u in sol.u]
I = [u[2] for u in sol.u]
R = [u[3] for u in sol.u]

plot(t, S, label="Susceptible", xlabel="Time", ylabel="Population", lw=2)
plot!(t, I, label="Infected", lw=2)
plot!(t, R, label="Recovered", lw=2)

true_vals = hcat(S,I,R)'  # Transposed it to make it comptabile in shape during element wise operations

# Define the time points corresponding to the predictions
time_points = 0:2:160  # From t=0 to t=160 with steps of 2

# STEP 2: SETTING UP THE NEURAL ODE
rng = Random.default_rng()
dudt2 = Lux.Chain(
    Lux.Dense(3, 64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, 3)
)

p, st = Lux.setup(rng, dudt2)

# Define NeuralODE with adjoint method
neural_ode = NeuralODE(dudt2, tspan, Tsit5(); saveat=2)

# Function to predict with the neural ODE
function predict_neuralode(p)
    sol2, _ = neural_ode(u0, p, st)  # Solve for (S, I, R)
    return hcat(sol2.u...)  # Extract the solution states and convert to a matrix
end

# Function to compute the loss 
function loss_neuralode(p)
    pred = predict_neuralode(p)  # Get the predictions
    loss = sum(abs2, true_vals .- pred)  # Compute the loss 
    return loss  # Return loss
end

# Define the callback function to track progress and plot predictions
callback = function(p, l; doplot = true)
    println("Loss: ", l)  # Print the loss
    if doplot
        pred = predict_neuralode(p)  # Get the current predictions
        plt = plot(time_points, true_vals[1, :]; label = "True S", xlabel="Time", ylabel="Population", lw=2)
        plot!(plt, time_points, pred[1, :]; label = "Predicted S", lw=2, linestyle=:dash)
        plot!(plt, time_points, true_vals[2, :]; label = "True I", lw=2)
        plot!(plt, time_points, pred[2, :]; label = "Predicted I", lw=2, linestyle=:dash)
        plot!(plt, time_points, true_vals[3, :]; label = "True R", lw=2)
        plot!(plt, time_points, pred[3, :]; label = "Predicted R", lw=2, linestyle=:dash)
        display(plot(plt))
    end
    return false  
end

#test
callback(p, loss_neuralode(p); doplot = true)

# TRAINING

pinit = ComponentArray(p)
adtype = Optimization.AutoZygote()

# Define the optimization function
optf = Optimization.OptimizationFunction((x, _) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

lr = 0.01  # Initial learning rate

for epoch in 1:4  # Run for a few epochs
    global result_neuralode  
    
    # Solve the optimization problem
    result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(lr); callback = callback, maxiters = 1000)
    
    # Update the optimization problem with the latest parameters after each epoch
    global optprob  
    optprob = remake(optprob; u0 = result_neuralode.u)
    
    global lr  
    lr *= 0.1  # Reduce the learning rate by a factor of 10 each epoch
end




# Step 2: Refine with BFGS optimizer
# Use the result from Adam training as the starting point for BFGS
###########optprob_bfgs = remake(optprob; u0 = result_neuralode.u)

##########result_neuralode2 = Optimization.solve(optprob_bfgs, Optim.LBFGS(); callback, allow_f_increases = true,maxiters = 5000)

#########callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)