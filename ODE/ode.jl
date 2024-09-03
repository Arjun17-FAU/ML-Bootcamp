using Plots
using DifferentialEquations

#Parameters
l = 1.0 #Length of Pendullum
g = 9.81 #Gravitational acceleration
m = 1.0

function pendullum!(du, u, p, t)
    du[1] = u[2]
    du[2] = ((-3*g/2*l)* sin(u[1])) + (3/m*l^2)*1 #Assuming external torque is of unit strength
end

uo = [0.01; 0.0]
tspan = (0,10)

prob = ODEProblem(pendullum!, uo, tspan)

sol = solve(prob)

plot(sol, idxs =(0,2)) #Angular displacement vs time
