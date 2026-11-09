# GPU variant of repro.jl (adaptive kernel only, plus uniform for reference).
# usage: julia --project=env_gpu repro_gpu.jl LIB FIDX SEED MODE T
#   MODE = cpu      : identical to repro.jl adaptive part (CPU reference with this T)
#   MODE = gpu      : kernel Gram matrices and the leave-one-out loss are evaluated on the GPU during training,
#                     optimizer (IPNewton, Zygote+ForwardDiff Hessian) unchanged
#   MODE = gpuhybrid: train on CPU (as MODE=cpu), evaluate the final estimator on the GPU
using JLD2, LinearAlgebra, Statistics, Random, GenericLinearAlgebra, SpecialFunctions, ArrayAllocators,
      KernelFunctions, ChainRules, Functors, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq,
      Suppressor, SciMLSensitivity, Optimization, OptimizationOptimJL, ForwardDiff, Lebedev, CUDA, cuDNN
LIB = ARGS[1]
include(joinpath(LIB, "PCNK", "src", "PCNK.jl")); using .PCNK
FIDX = parse.(Int, split(ARGS[2], ",")); SEED = parse(Int, ARGS[3]); MODE = ARGS[4]
const T = ARGS[5] == "Float32" ? Float32 : Float64
println("MODE=", MODE, " T=", T, " device=", CUDA.functional() ? CUDA.name(CUDA.device()) : "none")
dev = MODE == "cpu" ? identity : gpu_device()
data = jldopen(joinpath(LIB, "400ms", "SPM.jld"), "r") do fh; Dict(k => read(fh, k) for k in keys(fh)); end
freqs = data["freqs"]; c = T(343); ks = T.(2*π*freqs/c)
Random.seed!(SEED)
ir0 = data["recordings"]; X = T.(data["xyz_rec"])
ir = ir0 + 10^(-20/20) * std(ir0, dims = 1) .* randn(Complex{Float64}, size(X, 2), length(ks))  # same noise draw as repro.jl
ir = Complex{T}.(ir); ir_val = Complex{T}.(data["validation"]); X_val = T.(data["xyz_val"])
λ = T(0.011)
W = Chain(Dense(3,5, tanh), Dense(5,5, tanh), Dense(5,1,tanh), softplus)
κ_prop = PCNK.DirectedResidualPINKernel{T}(ks[2], 5, 11, W)
Xd = dev(X)
function ℓ(x, yy, κ, ps)
    kernel = dev(κ(ps)); K = kernel(x,x)
    reg_n = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.σ')
    reg_a = sum(kernel.AnalyticalKernel.σ)
    K_inv = inv(K + λ*(reg_a + reg_n)*I); α = K_inv*yy
    return sum(abs2, diag(K_inv) .\ α) + T(0.5)*reg_n
end
cons!(constraint, χ, _) = (constraint .= [sum(χ[1:14]); χ[1:28]])
lcons = T[1; fill(zero(T), 28)]; ucons = T[1; fill(T(100), 28)]
for f in FIDX
    y = ir[:, f]; y_val = ir_val[:, f]
    κ_prop.AnalyticalKernel.k = ks[f]; κ_prop.NeuralKernel.k = ks[f]
    χ, κ = destructure(κ_prop)
    trainmode = MODE == "gpu" ? dev : identity
    yd = trainmode(y); Xt = trainmode(X)
    t_train = @elapsed begin
        sol = copy(χ)
        func = Optimization.OptimizationFunction((χ_, yy)->ℓ(Xt, yd, κ, χ_), Optimization.AutoZygote(), cons = cons!)
        prob = Optimization.OptimizationProblem(func, χ, y, lcons = lcons, ucons = ucons)
        sol .= Optimization.solve(prob, IPNewton(), maxiters=200; f_calls_limit=500)
    end
    χa = T.(sol); χa[abs.(χa) .< eps(T)] .= zero(T)
    κe = κ(χa)
    reg = sum(κe.NeuralKernel.W(κe.NeuralKernel.v) .* κe.NeuralKernel.σ') + sum(κe.AnalyticalKernel.σ)
    t_eval = @elapsed begin
        κd = dev(κe); Xe = dev(X); Xv = dev(X_val)
        α = (κd(Xe,Xe)+λ*reg*I) \ dev(y)
        yh = Array(κd(Xv, Xe)*α)
    end
    e = 10log10(sum(abs2, y_val - yh)/sum(abs2, y_val))
    println("RESULT mode=", MODE, " T=", T, " f=", freqs[f], " Hz adaptive NMSE_dB=", round(e,digits=2), " train_s=", round(t_train,digits=1), " eval_s=", round(t_eval,digits=2))
end
