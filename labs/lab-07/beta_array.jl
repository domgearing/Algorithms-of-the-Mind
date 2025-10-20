struct BetaArray <: Gen.Distribution{Array{Float64}} end
const beta_array = BetaArray()

function Gen.logpdf(::BetaArray, x::Array{Float64}, a, b)
    @assert length(x) == length(a) == length(b) "Length missmatch"
    n = length(x)
    logprob = 0.0
    @inbounds for i = 1:n
        logprob += Gen.logpdf(Gen.beta, x[i], a[i], b[i])
    end
    return logprob
end

function Gen.random(::BetaArray, a, b)
    @assert length(a) == length(b) "Length missmatch"
    n = length(a)
    result = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        result[i] = Gen.random(Gen.beta, a[i], b[i])
    end
    return result
end

function Gen.logpdf_grad(::BetaArray, x::Array{Float64}, a, b)
    @assert length(x) == length(a) == length(b) "Length missmatch"
    n = length(x)
    xgrads = Vector{Float64}(undef, n)
    agrads = Vector{Float64}(undef, n)
    bgrads = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        xgrads[i], agrads[i], bgrads[i] = Gen.logpdf_grad(Gen.beta, x[i], a[i], b[i])
    end
    (xgrads, agrads, bgrads)
end

Gen.has_output_grad(::BetaArray) = true
Gen.has_argument_grads(::BetaArray) = (true, true)

export beta_array