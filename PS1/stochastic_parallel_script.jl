using Distributed
addprocs(2)
@everywhere using Parameters, Plots, SharedArrays, LinearAlgebra #import the libraries we want
include("stochastic_parallel_functions.jl") #import the functions that solve our growth model
@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!

@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
plot(k_grid, val_func[:, 1],ylabel = "V(K,z)", label = "Z=z_g", xlabel = "K", color="blue")
plot!(k_grid, val_func[:, 2],ylabel = "V(K,z)", label = "Z=z_b", xlabel = "K", color="orange")
savefig("PS1/results/Value_Functions_sp.png")

#policy functions
plot(k_grid, pol_func[:, 1],ylabel = "K'(K, z)", label = "Z=z_g",xlabel = "K",color="blue", linestyle=:solid)
plot!(k_grid, pol_func[:, 2],ylabel = "K'(K, z)", label = "Z=z_b",xlabel = "K",color="orange", linestyle=:solid)
plot!(k_grid,k_grid,label = "45 degree",color="red",linestyle=:dash)
savefig("PS1/results/Policy_Functions_sp.png")

#changes in policy function
pol_func_δ = pol_func .- reshape(k_grid, :, 1)
plot(k_grid, pol_func_δ[:, 1],
	ylabel = "K'(K,z) - K", label = "Z=z_g", xlabel = "K", color="blue")
plot!(k_grid, pol_func_δ[:, 2], label = "Z=z_b", xlabel = "K", color="orange")
hline!([0], linestyle=:dash, color=:black, label="")
savefig("PS1/results/Policy_Functions_Changes_sp.png")

#value function differences across adjacent capital grid points
value_diff = val_func[2:end, :] .- val_func[1:end-1, :]
plot(k_grid[1:end-1], value_diff[:, 1],
	ylabel = "V(K',z) - V(K,z)", label = "Z=z_g", xlabel = "K", color="blue")
plot!(k_grid[1:end-1], value_diff[:, 2], label = "Z=z_b", color="orange")
hline!([0], linestyle=:dash, color=:black, label="")
savefig("PS1/results/Value_Function_Differences_sp.png")

println("All done!")
################################
