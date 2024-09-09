#Some ODE to learn

#1 Radioactive decay

using Plots
using DifferentialEquations

a = 5.730 #Decay constant or Half life of Carbon 14

f(N,p,t) = -a*N

No = 1.0
tspan = (0.0,1.0)

prob = ODEProblem(f,No,tspan)

sol = solve(prob, Tsit5())

plot(sol, title ="How the number of carbon-14 change with time", xaxis ="Time(t)", yaxis = "N(t)")


#2 Simple Harmonic Oscillator: Linear Second Order ODE

w = 1.0

 #Since we need position x and angular velocity dx to define such a second order problem, we use a diffferent approach to define the function

function oscillator(dxx, dx, x, w, t)
    dxx .= -w*x
end

dxo = [π / 2]
xo = [0.0]
tspan = (0.0, 2π)

prob1 = SecondOrderODEProblem(oscillator, dxo, xo, tspan, w)

sol1 = solve(prob1, DPRKN6())

typeof(sol1)
size(sol1) #Shape is 2x22 (2 variables position x and angular velocity v is detrmined at 22 time points)

plot(sol1, idxs = [2,1], xaxis ="Time", yaxis = "Position")


#3 Second Order Non-Linear ODE - Simple Pendullum
# Define constants
g = 9.81  # Acceleration due to gravity
l = 1.0   # Length of the pendulum

# Define the second-order ODE for the pendulum
function Pend(dyy, dy, y, p, t)
    dyy .= -(g/l) .* sin.(y)
end

# Initial conditions: initial angular velocity and displacement
dyo = [0.0]         # Initial angular velocity (rad/s)
yo = [π / 2]        # Initial angular displacement (rad)
tspan = (0.0, 10.0) # Time span for the simulation

# Define the ODE problem
prob2 = SecondOrderODEProblem(Pend, dyo, yo, tspan)

# Solve the problem using a solver suitable for second-order ODEs
sol2 = solve(prob2, DPRKN6())

# Plot the results
plot(sol2, idxs=[2,1], title="Pendulum Motion", xlabel="Time (s)", ylabel="Angle/Angular Velocity", label=["θ(t)" "ω(t)"])


#4 Simple Pendullum
# In the above case, we solved it as a second order ODE. However, we can split it into a coupled set of first order ODE and solve it as shown below

function PendasFirstorder(du,u,p,t)
    du[1] = u[2]
    du[2] = -(g/l)*sin(u[1])
end

uo = [0, π / 2]
tspan = (0.0, 6.3)

prob4 = ODEProblem(PendasFirstorder, uo, tspan)

sol4 = solve(prob4, Tsit5())

plot(sol4, xaxis = "Time", yaxis = "Height")