# Library for 1D Gaussian messages and distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

import Statistics.mean

"""
Data structure that captures the state of an normalized 1D Gaussian. 
In this represenation, we are storing the precision times mean (τ) and the 
precision (ρ). This representations allows for numerically stable products of 
1D-Gaussians.
"""
struct Gaussian1D
    τ::Float64            # the precision mean, τ = μ/σ^2 = μρ, is the precision adjusted mean
    ρ::Float64            # the precision, ρ = 1/σ^2, is the inverse of the variance

    # default constructor checking for precision to be non-negative
    Gaussian1D(τ, ρ) =
        (ρ < 0) ? error("precision of a Gaussian must be non-negative") :
        new(promote(τ, ρ)...)
end
# Initializes a standard Gaussian 
Gaussian1D() = Gaussian1D(0, 1)

"""
    Gaussian1Dμσ2(μ,σ2)

Initializes a Gaussian from mean and variance.
"""
Gaussian1Dμσ2(μ, σ2) = Gaussian1D(μ / σ2, 1 / σ2)

"""
    show(io,g)

Pretty-prints a 1D Gaussian
"""
function Base.show(io::IO, g::Gaussian1D)
    if (g.ρ == 0.0)
        print(io, "μ = 0, σ = Inf")
    else
        print(io, "μ = ", mean(g), ", σ = ", sqrt(variance(g)))
    end
end

"""
    mean(g)

Returns the mean of the 1D-Gaussian
```julia-repl
julia> mean(Gaussian1D(1,2))
0.5

julia> mean(Gaussian1Dμσ2(1,2))
1.0
```
"""
mean(g::Gaussian1D) = g.τ / g.ρ

"""
    variance(g)

Returns the variance of the 1D-Gaussian 
```julia-repl
julia> variance(Gaussian1D(1,2))
0.5

julia> variance(Gaussian1Dμσ2(1,2))
2.0
```
"""
variance(g::Gaussian1D) = 1.0 / g.ρ


"""
    absdiff(g1,g2)

Computes the absolute difference of `g1` and `g2` in terms of τ and ρ
# Examples
```julia-repl
julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,2))
1.0

julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,3))
1.4142135623730951
```
"""
absdiff(g1::Gaussian1D, g2::Gaussian1D) = max(abs(g1.τ - g2.τ), sqrt(abs(g1.ρ - g2.ρ)))

"""
    *(g1,g2)

Multiplies two 1D Gaussians together and re-normalizes them
# Examples
```julia-repl
julia> Gaussian1D() * Gaussian1D()
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.:*(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.τ + g2.τ, g1.ρ + g2.ρ))
end

"""
    /(g1,g2)

Divides two 1D Gaussians from each other
# Examples
```julia-repl
julia> Gaussian1D(0,1) / Gaussian1D(0,0.5)
μ = 0.0, σ = 1.4142135623730951
```
"""
function Base.:/(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.τ - g2.τ, g1.ρ - g2.ρ))
end

"""
    logNormProduct(g1,g2)

Computes the log-normalization constant of a multiplication of `g1` and `g2`
# Examples
```julia-repl
julia> logNormProduct(Gaussian1D() * Gaussian1D())
c = 0.28209479177387814
```
"""
function logNormProduct(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.ρ == 0.0 || g2.ρ == 0.0)
        return (0.0)
    else
        σ2Sum = variance(g1) + variance(g2)
        μDiff = mean(g1) - mean(g2)
        return (-0.5 * (log(2 * π * σ2Sum) + μDiff * μDiff / σ2Sum))
    end
end

"""
    logNormRatio(g1,g2)

Computes the log-normalization constant of a division of `g1` with `g2`
# Examples
```julia-repl
julia> logNormRatio(Gaussian1D(0,1) / Gaussian1D(0,0.5))
5.013256549262001
```
"""
function logNormRatio(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.ρ == 0.0 || g2.ρ == 0.0)
        return (0.0)
    else
        g2Var = variance(g2)
        σ2Diff = g2Var - variance(g1)
        if (σ2Diff == 0.0)
            return (0.0)
        else
            μDiff = mean(g1) - mean(g2)
            return (log(g2Var) + 0.5 * (log(2 * π / σ2Diff) + μDiff * μDiff / σ2Diff))
        end
    end
end