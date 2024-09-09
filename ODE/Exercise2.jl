using Plots
using DifferentialEquations


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