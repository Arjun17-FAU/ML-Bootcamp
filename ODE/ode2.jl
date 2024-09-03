using DifferentialEquations
using Plots

# Constants
l = 1.0      # Length of the pendulum (m)
m = 1.0      # Mass of the pendulum (kg)
g = 9.81     # Gravitational acceleration (m/s^2)

# Define the system of ODEs
function pendulum!(du, u, p, t)
    θ, ω = u  # u[1] = θ(t), u[2] = ω(t)
    M = 0.0   # External torque, assume zero if not specified

    du[1] = ω
    du[2] = -(3g / (2l)) * sin(θ) + (3 / (m * l^2)) * M
end

# Initial conditions
θ0 = 0.01  # Initial angular deflection (rad)
ω0 = 0.0   # Initial angular velocity (rad/s)
u0 = [θ0, ω0]

# Time span
tspan = (0.0, 10.0)

# Define the ODE problem
prob = ODEProblem(pendulum!, u0, tspan)

# Solve the ODEs
sol = solve(prob)

# Extract θ(t) and ω(t) for phase space plotting
θ = sol[1,:]  # Angular displacement over time
ω = sol[2,:]  # Angular velocity over time

# Plot the phase space (ω vs θ)
plot(θ, ω, xlabel="θ (Angular Displacement)", ylabel="ω (Angular Velocity)", title="Phase Space Plot")
print(sol[1,:])