# Command-line bridge to PCNK's adaptive kernel (DirectedResidualPINKernel), one problem per call.
# usage: julia --project=<env dir> bridge.jl in.npz out.npz
# in.npz : x (M,3) mic positions [m], y (M,) complex pressure, q (N,3) targets [m], f (Hz), c (m/s), seed
# out.npz: p (N,) complex predictions, plus fit diagnostics.
# Training uses the library's own schedule (IPNewton, second order) on CPU; there is no GPU path (see README).
using NPZ, LinearAlgebra, Random, Statistics, GenericLinearAlgebra, SpecialFunctions, ArrayAllocators,
      KernelFunctions, ChainRules, JLD2, Functors, Optimisers, FFTW, Flux, KernelAbstractions,
      OrdinaryDiffEq, Suppressor, SciMLSensitivity, Optimization, OptimizationOptimJL, ForwardDiff, Lebedev
const LIB = get(ENV, "PCNK_DIR", joinpath(@__DIR__, "..", "..", "PhysicsConstrainedNeuralKernel"))
include(joinpath(LIB, "PCNK", "src", "PCNK.jl"))
using .PCNK

function main(inpath, outpath)
    d = npzread(inpath)
    T = Float64
    X = Matrix{T}(permutedims(d["x"]))          # library convention: 3 x M (columns are points)
    Q = Matrix{T}(permutedims(d["q"]))
    y = Vector{Complex{T}}(d["y"])
    f = T(only(d["f"])); c = T(only(d["c"])); seed = Int(only(d["seed"]))
    k = 2π*f/c
    Random.seed!(seed)
    # Hyperparameters and training schedule exactly as in the library's SPM.jl example (adaptive kernel):
    Ord_dir = 5; Ord_NN = 11; λ = 0.011
    W = Chain(Dense(3,5, tanh), Dense(5,5, tanh), Dense(5,1,tanh), softplus)
    κ_prop = PCNK.DirectedResidualPINKernel{T}(k, Ord_dir, Ord_NN, W)
    function ℓ(x, yy, κ, ps)            # closed-form leave-one-out cross-validation loss + 0.5*reg_n
        kernel = κ(ps); K = kernel(x,x)
        reg_n = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.σ')
        reg_a = sum(kernel.AnalyticalKernel.σ)
        K_inv = inv(K + λ*(reg_a + reg_n)*I); α = K_inv*yy
        return sum(abs2, diag(K_inv) .\ α) + 0.5*reg_n
    end
    cons!(constraint, χ, _) = (constraint .= [sum(χ[1:14]); χ[1:28]])
    lcons = T[1; fill(zero(T), 28)]; ucons = T[1; fill(T.(100), 28)]
    χ, κ = destructure(κ_prop)
    t_fit = @elapsed begin
        sol = copy(χ)
        @suppress begin
            func = Optimization.OptimizationFunction((χ_, yy)->ℓ(X, yy, κ, χ_), Optimization.AutoZygote(), cons = cons!)
            prob = Optimization.OptimizationProblem(func, χ, y, lcons = lcons, ucons = ucons)
            sol .= Optimization.solve(prob, IPNewton(), maxiters=200; f_calls_limit=500)
        end
    end
    χ = T.(sol); χ[abs.(χ) .< eps(T)] .= zero(T)
    κe = κ(χ)
    reg = sum(κe.NeuralKernel.W(κe.NeuralKernel.v) .* κe.NeuralKernel.σ') + sum(κe.AnalyticalKernel.σ)
    K = κe(X, X)
    α = (K + λ*reg*I) \ y
    p = κe(Q, X) * α
    fit_nmse = sum(abs2, κe(X,X)*α - y)/sum(abs2, y)
    out = Dict("p" => Vector{ComplexF64}(p), "fit_seconds" => [t_fit], "train_residual" => [fit_nmse])
    npzwrite(outpath, out)
end
main(ARGS[1], ARGS[2])
