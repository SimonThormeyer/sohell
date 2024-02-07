# Library for collections of distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

include("collections.jl")

using Distributions

"""
Data structure that a factor
"""
struct Factor
    update!::Function        # the update function for a factor to its related variables
    logVarNorm!::Function    # Computes the log normalization for all the neighbouring variables of the factor from the messages
    logFactorNorm::Function  # Computes the log-normalization constant for factors
    noMessages::Int64        # number of messages
end

"""
    GaussianFactor(g, idx, bag)

Creates a new factor `bag[idx]` = `g` 
```
"""
function GaussianFactor(g::Gaussian1D, idx::Int, bag::DistributionBag)
    msg = Gaussian1D(0, 0)

    function update!(i)
        if (i == 1)
            oldMarginal = bag[idx]
            newMarginal = oldMarginal / msg * g
            bag[idx] = newMarginal
            msg = g
            return (absdiff(oldMarginal, newMarginal))
        else
            throw(ArgumentError("Gaussian factor can only update variable 1"))
        end
    end

    function logVarNorm!()
        logZ = logNormProduct(bag[idx], msg)
        bag[idx] *= msg
        return (logZ)
    end

    return (Factor(update!, logVarNorm!, () -> 0, 1))
end

"""
    GaussianFactor(μ, σ2, idx, bag)

Creates a new factor N(bag[idx];μ,σ2) of a 1D Gaussian with mean `μ` and variance `σ2`
for the variable `bag[idx]`
```
"""
GaussianFactor(μ, σ2, idx, bag) = GaussianFactor(Gaussian1Dμσ2(μ, σ2), idx, bag)

"""
    GaussianMeanFactor(β2, idx1, idx2, bag)

Creates a new factor N(bag[idx1];bag[idx2],β2) of a 1D Gaussian with mean `bag[idx2]` 
and variance `β2` for the variable `bag[idx1]`
"""
function GaussianMeanFactor(β2, idx1::Int, idx2::Int, bag::DistributionBag)
    return GaussianMeanFactor(β2, idx1, idx2, 0.0, bag)
end

"""
GaussianMeanFactor(β2, idx1, idx2, κ, bag)

Creates a new factor N(bag[idx1];bag[idx2] + κ, β2) of a 1D Gaussian with mean `bag[idx2]` + offset `κ` 
and variance `β2` for the variable `bag[idx1]`
"""
function GaussianMeanFactor(β2, idx1::Int, idx2::Int, κ::Float64, bag::DistributionBag)
    msg1 = Gaussian1D(0, 0)
    msg2 = Gaussian1D(0, 0)

    function update!(i)
        if (i == 1)
            msgBack = bag[idx2] / msg2

            # This is the stable version of        
            #   newMsg = Gaussian1Dμσ2(mean(msgBack), variance(msgBack) + β2)
            c = 1 / (1 + β2 * msgBack.ρ)
            τ′ = c * (msgBack.τ + msgBack.ρ * κ * sqrt(β2))
            newMsg = Gaussian1D(τ′, c * msgBack.ρ)
            oldMarginal = bag[idx1]
            newMarginal = oldMarginal / msg1 * newMsg
            bag[idx1] = newMarginal
            msg1 = newMsg
            return (absdiff(oldMarginal, newMarginal))
        elseif (i == 2)
            msgBack = bag[idx1] / msg1

            # This is the stable version of        
            #   newMsg = Gaussian1Dμσ2(mean(msgBack), variance(msgBack) + β2)
            c = 1 / (1 + β2 * msgBack.ρ)
            τ′ = c * (msgBack.τ - msgBack.ρ * κ * sqrt(β2))
            newMsg = Gaussian1D(τ′, c * msgBack.ρ)

            oldMarginal = bag[idx2]
            newMarginal = oldMarginal / msg2 * newMsg
            bag[idx2] = newMarginal
            msg2 = newMsg
            return (absdiff(oldMarginal, newMarginal))
        else
            throw(ArgumentError("Gaussian factor can only update variable 1 or 2"))
        end
    end

    function logVarNorm!()
        logZ = logNormProduct(bag[idx1], msg1)
        logZ += logNormProduct(bag[idx2], msg2)
        bag[idx1] *= msg1
        bag[idx2] *= msg2
        return (logZ)
    end

    function logFactorNorm()
        logZ = logNormRatio(bag[idx2], msg2)
        return (logZ)
    end

    return (Factor(update!, logVarNorm!, logFactorNorm, 2))
end

"""
    WeightedSumFactor(a1, a2, idx1, idx2, idx3, bag)

Creates a new factor I(bag[idx3] = a1 * bag[idx1] + a2 * bag[idx2]) for the three 
variables `bag[idx1]`, `bag[idx2]` and `bag[idx3]`
"""
function WeightedSumFactor(a1, a2, idx1::Int, idx2::Int, idx3::Int, bag::DistributionBag)
    msg1 = Gaussian1D(0, 0)
    msg2 = Gaussian1D(0, 0)
    msg3 = Gaussian1D(0, 0)

    function update!(i)
        if (i == 1)
            msgBack2 = bag[idx2] / msg2
            msgBack3 = bag[idx3] / msg3

            newMsg1 = Gaussian1Dμσ2(
                mean(msgBack3) / a1 - a2 / a1 * mean(msgBack2),
                variance(msgBack3) / (a1 * a1) + a2 * a2 / (a1 * a1) * variance(msgBack2),
            )

            oldMarginal = bag[idx1]
            newMarginal = oldMarginal / msg1 * newMsg1
            bag[idx1] = newMarginal
            msg1 = newMsg1
            return (absdiff(oldMarginal, newMarginal))
        elseif (i == 2)
            msgBack1 = bag[idx1] / msg1
            msgBack3 = bag[idx3] / msg3

            newMsg2 = Gaussian1Dμσ2(
                mean(msgBack3) / a2 - a1 / a2 * mean(msgBack1),
                variance(msgBack3) / (a2 * a2) + a1 * a1 / (a2 * a2) * variance(msgBack1),
            )

            oldMarginal = bag[idx2]
            newMarginal = oldMarginal / msg2 * newMsg2
            bag[idx2] = newMarginal
            msg2 = newMsg2
            return (absdiff(oldMarginal, newMarginal))
        elseif (i == 3)
            msgBack1 = bag[idx1] / msg1
            msgBack2 = bag[idx2] / msg2

            newMsg3 = Gaussian1Dμσ2(
                a1 * mean(msgBack1) + a2 * mean(msgBack2),
                a1 * a1 * variance(msgBack1) + a2 * a2 * variance(msgBack2),
            )

            oldMarginal = bag[idx3]
            newMarginal = oldMarginal / msg3 * newMsg3
            bag[idx3] = newMarginal
            msg3 = newMsg3
            return (absdiff(oldMarginal, newMarginal))
        else
            throw(
                ArgumentError(
                    "Gaussian likelihood factor can only update variable 1, 2 or 3",
                ),
            )
        end
    end

    function logVarNorm!()
        logZ = logNormProduct(bag[idx1], msg1)
        logZ += logNormProduct(bag[idx2], msg2)
        logZ += logNormProduct(bag[idx3], msg3)
        bag[idx1] *= msg1
        bag[idx2] *= msg2
        bag[idx3] *= msg3
        return (logZ)
    end

    function logFactorNorm()
        logZ = logNormRatio(bag[idx1], msg1) + logNormRatio(bag[idx2], msg2)
        return (logZ)
    end

    return (Factor(update!, logVarNorm!, logFactorNorm, 3))
end

"""
    GreaterThanFactor(ϵ, idx, bag)

Creates a new factor I(bag[idx] ≥ ϵ) for the variable `bag[idx]`
"""
function GreaterThanFactor(ϵ, idx::Int, bag::DistributionBag)
    d = Normal()

    # computes the additive correction of a single-sided truncated Gaussian with unit variance
    function v(t, ϵ)
        denom = cdf(d, t - ϵ)
        if (denom < floatmin(Float64))
            return (ϵ - t)
        else
            return (pdf(d, t - ϵ) / denom)
        end
    end

    # computes the multiplicative correction of a single-sided truncated Gaussian with unit variance
    function w(t, ϵ)
        denom = cdf(d, t - ϵ)
        if (denom < floatmin(Float64))
            return ((t - ϵ < 0.0) ? 1.0 : 0.0)
        else
            vt = v(t, ϵ)
            return (vt * (vt + t - ϵ))
        end
    end

    msg = Gaussian1D(0, 0)

    function update!(i)
        if (i == 1)
            msgBack = bag[idx] / msg
            a = msgBack.τ / sqrt(msgBack.ρ)
            b = ϵ * sqrt(msgBack.ρ)
            c = 1.0 - w(a, b)
            newMarginal = Gaussian1D(
                (msgBack.τ + sqrt(msgBack.ρ) * v(a, b)) / c,
                msgBack.ρ / c,
                # msgBack.logC + log(1 - cdf(d, b - a)),
            )
            oldMarginal = bag[idx]
            msg = newMarginal / msgBack
            bag[idx] = newMarginal
            return (absdiff(oldMarginal, newMarginal))
        else
            throw(ArgumentError("Greater-Than factor can only update variable 1"))
        end
    end

    function logVarNorm!()
        logZ = logNormProduct(bag[idx], msg)
        bag[idx] *= msg
        return (logZ)
    end

    function logFactorNorm()
        msgBack = bag[idx] / msg
        logZ =
            -logNormProduct(msgBack, msg) +
            log(cdf(d, (mean(msgBack) - ϵ) / sqrt(variance(msgBack))))
        return (logZ)
    end

    return (Factor(update!, logVarNorm!, logFactorNorm, 1))
end

"""
    logNormalization(factorList, bag)

Computes the log-normalization constant of the factor graph specified by `factorList` using the distribution collection `bag`
```julia-repl
julia> bag = DistributionBag(Gaussian1D(0, 0))
Prior: μ = 0, σ = Inf
  empty bag

julia> s = add!(bag)
1

julia> factorList = Vector{Factor}()
Factor[]

julia> function addFactor(f) push!(factorList, f); return(f); end
addFactor (generic function with 1 method)

julia> priorF = addFactor(GaussianFactor(Gaussian1Dμσ2(0,1),s,bag))
Factor(var"#update!#8"{Gaussian1D, Int64, DistributionBag}(μ = 0.0, σ = 1.0, 1, Prior: μ = 0, σ = Inf
  [1]: μ = 0, σ = Inf
, Core.Box(μ = 0, σ = Inf)), var"#logVarNorm!#9"{Int64, DistributionBag}(1, Prior: μ = 0, σ = Inf
  [1]: μ = 0, σ = Inf
, Core.Box(μ = 0, σ = Inf)), var"#7#10"(), 1)

julia> matchF = addFactor(GreaterThanFactor(0,s,bag))
Factor(var"#update!#19"{Int64, Int64, DistributionBag, var"#w#18"{var"#v#17"{Normal{Float64}}, Normal{Float64}}, var"#v#17"{Normal{Float64}}}(0, 1, Prior: μ = 0, σ = Inf
  [1]: μ = 0, σ = Inf
, Core.Box(μ = 0, σ = Inf), var"#w#18"{var"#v#17"{Normal{Float64}}, Normal{Float64}}(var"#v#17"{Normal{Float64}}(Normal{Float64}(μ=0.0, σ=1.0)), Normal{Float64}(μ=0.0, σ=1.0)), var"#v#17"{Normal{Float64}}(Normal{Float64}(μ=0.0, σ=1.0))), var"#logVarNorm!#20"{Int64, DistributionBag}(1, Prior: μ = 0, σ = Inf
  [1]: μ = 0, σ = Inf
, Core.Box(μ = 0, σ = Inf)), var"#logFactorNorm#21"{Int64, Int64, DistributionBag, Normal{Float64}}(0, 1, Prior: μ = 0, σ = Inf
  [1]: μ = 0, σ = Inf
, Core.Box(μ = 0, σ = Inf), Normal{Float64}(μ=0.0, σ=1.0)), 1)

julia> priorF.update!(1)
1.0

julia> priorF.update!(1)
0.0

julia> matchF.update!(1)
2.1957291567607653

julia> matchF.update!(1)
0.0

julia> exp(logNormalization(factorList, bag))
0.5
```
"""
function logNormalization(factorList::Vector{Factor}, bag::DistributionBag)
    # reset all the variables to the prior
    reset!(bag)

    logZ = 0
    # re-compute the marginals of all factors
    for i in eachindex(factorList)
        logZ += factorList[i].logVarNorm!()
    end

    # re-normalize the factors 
    for i in eachindex(factorList)
        logZ += factorList[i].logFactorNorm()
    end

    return (logZ)
end
