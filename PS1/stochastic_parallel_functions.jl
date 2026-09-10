#= 
This code is a parallelized version of the VFI code for the neoclassical growth model.
The main difference is that the Bellman operator is parallelized using the @distributed macro.
September 2024
=#


@everywhere @with_kw struct Primitives
    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36
    k_min::Float64 = 0.01
    k_max::Float64 = 90.0
    nk::Int64 = 1000
    k_grid::SharedVector{Float64} = collect(range(start=k_min, stop=k_max, length=nk))
    z_g = 1.25 # positive prod. shock
    z_b = 0.2 # negative prod. shock
    z_grid = [z_g, z_b]
    nz = 2
    Pi = [0.977 0.023; 0.074 0.926] # transition
end


@everywhere @with_kw mutable struct Results
    val_func::SharedMatrix{Float64}
    pol_func::SharedMatrix{Float64}
end


@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedMatrix{Float64}(zeros(prim.nk, prim.nz))
    pol_func = SharedMatrix{Float64}(zeros(prim.nk, prim.nz))
    res = Results(val_func, pol_func)
    prim, res
end


function Bellman(prim::Primitives, res::Results)
    @unpack_Results res
    @unpack_Primitives prim
    
    v_next = SharedMatrix{Float64}(zeros(nk, nz))

    #choice_lower = 1
    for z_index in 1:nz
        z = z_grid[z_index]
        @sync @distributed for k_index in 1:nk 
            k = k_grid[k_index]
            candidate_max = -Inf
            budget = z*k^α + (1-δ)*k
            
            for kp_index in 1:nk
                c = budget - k_grid[kp_index]
                if c > 0
                    expected_value = dot(
                                        Pi[z_index, :],
                                        val_func[kp_index, :]
                                        )
                    val = log(c) + β*expected_value
                    if val > candidate_max
                        candidate_max = val
                        res.pol_func[k_index, z_index] = k_grid[kp_index]
                        #choice_lower = kp_index
                    end
                end
            end
            v_next[k_index, z_index] = candidate_max
        end
    end

    return v_next
end


function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0

    while err > tol
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        n += 1
    end
    println("Value function converged in ", n, " iterations.")
end


#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end