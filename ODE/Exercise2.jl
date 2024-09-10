using Plots
using DifferentialEquations

#PROBLEM ONE: SIMPLE Pendullum

#Consider a vector u where we store the position and  angular velocity respectively

#Let us assume the pendullum is disturbed by an external time varying forcing function given by sin(2t)

#The vector du will store their gradients

l = 1.0 #m
g = 9.81
m  = 1.0

function SimplePendullumFirstOrder!(du,u,p,t)
    du[1] = u[2]
    du[2] = -(3*g/2*l)*sin(u[1]) + (3/m*l*l)*sin(2*t)
end

uo = [0.01, 0.0]
tspan = (0,10)
prob = ODEProblem(SimplePendullumFirstOrder!,uo,tspan)

sol = solve(prob, Tsit5())


p1 = plot(sol.t, sol[1,:], linewidth = 2, title = "Pendulum Position (θ) with external forcing", xaxis = "Time", yaxis = "Position (θ)", label = "θ(t)", legend=:topright)

display(p1)

p2 = plot(sol.t, sol[2,:], linewidth = 2, title = "Pendulum Velocity (ω) with external forcing", xaxis = "Time", yaxis = "Velocity (ω)", label = "ω(t)", legend=:topright)

display(p2)

p3 = plot(sol, title = "Pendulum with Time-Varying Force sin(2t)", xaxis = "Time", yaxis = "Angle/Angular Velocity", label = ["Position(t)" "Angular Velocity(t)"])
display(p3)

#No forcing function
function SimplePendullumFirstOrderNF!(dx,x,p,t)
    dx[1] = x[2]
    dx[2] = -(3*g/2*l)*sin(x[1]) + (3/m*l*l)*0
end

xo = [0.01, 0.0]
tspan = (0,10)
prob1 = ODEProblem(SimplePendullumFirstOrderNF!,xo,tspan)
sol1 = solve(prob1, Tsit5())

p4 = plot(sol1.t, sol1[1,:], linewidth = 2, title = "Pendulum Position (θ) without External forcing", xaxis = "Time", yaxis = "Position (θ)", label = "θ(t)", legend=:topright)

display(p4)

p5 = plot(sol1.t, sol1[2,:], linewidth = 2, title = "Pendulum Velocity (ω) without External forcing", xaxis = "Time", yaxis = "Velocity (ω)", label = "ω(t)", legend=:topright)

display(p5)

p6 = plot(sol1, title = "Pendulum without External Forcing", xaxis = "Time", yaxis = "Angle/Angular Velocity", label = ["Position(t)" "Angular Velocity(t)"])
display(p6)


p7 = plot(p1, p4, layout = (2, 1), size = (600, 800))
display(p7)


p8 = plot(p2, p5, layout = (2, 1), size = (600, 800))
display(p8)


#PROBLEM 2: SIR MODEL

#Parameters
beta = 0.3
gamma = 0.1
N = 1000

function SIR!(du, u, p, t)
    du[1] = -(beta*u[1]*u[2])/N
    du[2] =  ((beta*u[1]*u[2])/N) - gamma*u[2]
    du[3] =  gamma*u[2]
end

uo = [999, 1, 0 ]
tspan = (0,160)

prob2 = ODEProblem(SIR!, uo, tspan)
sol2 = solve(prob2, Tsit5())

p9 = plot(sol2.t, sol2[1,:], linewidth = 2, title = "Susceptible population over time", xaxis = "Time", yaxis = "No: of people", label = "S(t)", legend=:topright)

display(p9)

p10 = plot(sol2.t, sol2[2,:], linewidth = 2, title = "Infected population over time", xaxis = "Time", yaxis = "No: of people", label = "I(t)", legend=:topright)

display(p10)

p11 = plot(sol2.t, sol2[3,:], linewidth = 2, title = "Recovered population over time", xaxis = "Time", yaxis = "No: of people", label = "R(t)", legend=:topright)

display(p11)