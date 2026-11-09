# Reduced reproduction of PhysicsConstrainedNeuralKernel/SPM.jl (400 ms room).
# Same kernels, losses, optimizers and hyperparameters as SPM.jl; only differences:
#   * only the frequency indices in FIDX are run (SPM.jl runs all 11 bins)
#   * RNG seeded (SPM.jl is unseeded), no plotting, no pointwise-plane evaluation
using JLD2, LinearAlgebra, Statistics, Random, GenericLinearAlgebra, SpecialFunctions, ArrayAllocators,
      KernelFunctions, ChainRules, Functors, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq,
      Suppressor, SciMLSensitivity, Optimization, OptimizationOptimJL, ForwardDiff, Lebedev
LIB = ARGS[1]
include(joinpath(LIB, "PCNK", "src", "PCNK.jl"))
using .PCNK
const T = Float64
FIDX = parse.(Int, split(ARGS[2], ","))
Random.seed!(parse(Int, ARGS[3]))
# FileIO routes *.jld to the legacy JLD package, so read the file with JLD2.jldopen directly.
data = jldopen(joinpath(LIB, "400ms", "SPM.jld"), "r") do fh; Dict(k => read(fh, k) for k in keys(fh)); end
freqs = data["freqs"]; c = T(343); ks = T.(2*π*freqs/c)
SNR = 20; ir0 = data["recordings"]; X = data["xyz_rec"]
ir = ir0 + 10^(-SNR/20) * std(ir0, dims = 1) .* randn(Complex{T}, size(X, 2), length(ks))
ir_val = data["validation"]; X_val = data["xyz_val"]
println("freqs = ", freqs, "  M=", size(X,2), " Mval=", size(X_val,2))
Ord_dir = 5; Ord_NN = 11; λ = 0.011
κ_uni = PCNK.UniformKernel{T}(ks[2])
W = Chain(Dense(3,5, tanh), Dense(5,5, tanh), Dense(5,1,tanh), softplus)
κ_pcnk = PCNK.NeuralWeightPlaneWaveKernel{T}(ks[2], Ord_NN, W=W)
κ_prop = PCNK.DirectedResidualPINKernel{T}(ks[2], Ord_dir, Ord_NN, W)

function ℓ_pcnk(x, y, κ, ps)
    kernel = κ(ps); K = kernel(x,x)
    reg_n = sum(kernel.W(kernel.v) .* kernel.σ')
    K_inv = inv(K + λ*(reg_n)*I); α = K_inv*y
    return sum(abs2, diag(K_inv) .\ α)
end
function ℓ_prop(x, y, κ, ps)
    kernel = κ(ps); K = kernel(x,x)
    reg_n = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.σ')
    reg_a = sum(kernel.AnalyticalKernel.σ)
    K_inv = inv(K + λ*(reg_a + reg_n)*I); α = K_inv*y
    return sum(abs2, diag(K_inv) .\ α) + 0.5*reg_n
end
function cons!(constraint, χ, y); constraint .= [sum(χ[1:14]); χ[1:28]]; end
lcons = T[1; fill(zero(T), 28)]; ucons = T[1; fill(T.(100), 28)]

for f in FIDX
    y = ir[:, f]; y_val = ir_val[:, f]
    nm(yh) = 10log10(sum(abs2, y_val - yh)/sum(abs2, y_val))
    # uniform
    κ_uni.k = ks[f]; K = κ_uni(X,X); α = (K + λ*I)\y
    e_uni = nm(κ_uni(X_val, X)*α)
    # PCNN (trainable residual only), LBFGS
    κ_pcnk.k = ks[f]; χ, κ = destructure(κ_pcnk)
    t_pcnk = @elapsed begin
        sol = copy(χ)
        @suppress begin
            func = Optimization.OptimizationFunction((χ,y)->ℓ_pcnk(X,y,κ,χ), Optimization.AutoZygote())
            prob = Optimization.OptimizationProblem(func, χ, y)
            sol .= Optimization.solve(prob, LBFGS(), maxiters=200; f_calls_limit=500)
        end
        χp = T.(sol); χp[abs.(χp) .< eps(T)] .= zero(T)
        κe = κ(χp); reg = sum(κe.W(κe.v) .* κe.σ'); K = κe(X,X)
        α = (K+λ*reg*I)\y; e_pcnk = nm(κe(X_val, X)*α)
    end
    # Adaptive kernel (directed + residual), IPNewton with constraints
    κ_prop.AnalyticalKernel.k = ks[f]; κ_prop.NeuralKernel.k = ks[f]
    χ, κ = destructure(κ_prop)
    t_prop = @elapsed begin
        sol = copy(χ)
        @suppress begin
            func = Optimization.OptimizationFunction((χ,y)->ℓ_prop(X,y,κ,χ), Optimization.AutoZygote(), cons = cons!)
            prob = Optimization.OptimizationProblem(func, χ, y, lcons = lcons, ucons=ucons)
            sol .= Optimization.solve(prob, IPNewton(), maxiters=200; f_calls_limit=500)
        end
        χa = T.(sol); χa[abs.(χa) .< eps(T)] .= zero(T)
        κe = κ(χa)
        reg = sum(κe.NeuralKernel.W(κe.NeuralKernel.v) .* κe.NeuralKernel.σ') + sum(κe.AnalyticalKernel.σ)
        K = κe(X,X); α = (K+λ*reg*I)\y; e_prop = nm(κe(X_val, X)*α)
    end
    println("RESULT f=", freqs[f], " Hz  NMSE_dB uniform=", round(e_uni,digits=2), " PCNN=", round(e_pcnk,digits=2),
            " adaptive=", round(e_prop,digits=2), "  train_s pcnn=", round(t_pcnk,digits=1), " adaptive=", round(t_prop,digits=1))
end
