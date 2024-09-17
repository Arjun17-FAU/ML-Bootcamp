using OrdinaryDiffEq
using DomainSets
using ModelingToolkit
using MethodOfLines
using Plots

# Parameters and variables
@parameters t x
@variables u_r(..) u_i(..)

# Define the operators
Dt = Differential(t)
Dxx = Differential(x)^2

# Real and imaginary parts of the Schrödinger equation (V(x) = 0)
eq_r = Dt(u_r(t,x)) ~ Dxx(u_i(t,x))
eq_i = -Dt(u_i(t,x)) ~ Dxx(u_r(t,x))

# Initial and boundary conditions
bcs = [u_r(0,x) ~ sin(2*π*x), u_i(0,x) ~ 0, u_r(t,0) ~ 0, u_r(t,1) ~ 0, u_i(t,0) ~ 0, u_i(t,1) ~ 0]

# Domain
domains = [t ∈ Interval(0.0, 1.0), x ∈ Interval(0.0, 1.0)]

# Define the PDE system
@named pdesys = PDESystem([eq_r, eq_i], bcs, domains, [t, x], [u_r(t,x), u_i(t,x)])

# Discretization step size
dx = 0.1
discretization = MOLFiniteDifference([x => dx], t)

# Discretize the PDE system
prob = discretize(pdesys, discretization)

# Solve the system
sol = solve(prob, Tsit5(), saveat = 0.1)

# Extract discrete values
discrete_x = sol[x]
discrete_t = sol[t]
solution_r = sol[u_r(t,x)]
solution_i = sol[u_i(t,x)]

# Plot the real part of the solution
plt = plot()
for i in eachindex(discrete_t)
    plot!(discrete_x, solution_r[i, :], label = "Re(u), t = $(discrete_t[i])")
end

# Plot the imaginary part of the solution
for i in eachindex(discrete_t)
    plot!(discrete_x, solution_i[i, :], label = "Im(u), t = $(discrete_t[i])")
end

plt

plt2 = plot()
@gif for i in eachindex(discrete_t)
    plt2 = plot(discrete_x, solution_r[i, :], label = "Re(u), t = $(discrete_t[i])", lw=2, ylim=(-1.0, 1.0), xlabel="x", ylabel="u")
    plot!(discrete_x, solution_i[i, :], label = "Im(u), t = $(discrete_t[i])", lw=2)
end every 1