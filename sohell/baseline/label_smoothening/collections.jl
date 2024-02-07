# Library for collections of distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

include("gaussian.jl")

"""
Data structure that stores a colletion of 1D Gaussians
"""
struct DistributionBag
    prior::Gaussian1D       # the prior that is used for each new variable
    bag::Vector{Gaussian1D} # the actual resizeable vector that stores the 1D Gaussians

    # default constructor
    DistributionBag(prior::Gaussian1D) = new(prior, Vector{Gaussian1D}(undef, 0))
end

"""
    show(io,bag)

Pretty-prints a distribution bag
"""
function Base.show(io::IO, db::DistributionBag)
    println(io, "Prior: ", db.prior)
    if (length(db.bag) == 0)
        println(io, "  empty bag")
    else
        for i in eachindex(db.bag)
            println(io, "  [", i, "]: ", db.bag[i])
        end
    end
end

"""
    add(db)

Adds a new distribution to the distribution bag `db` and returns a unique index
```
"""
function add!(db::DistributionBag)
    push!(db.bag, db.prior)
    return (length(db.bag))
end

"""
    reset!(db)

Resets all distribution to the prior in the distribution bag `db`
```
"""
function reset!(db::DistributionBag)
    for i in eachindex(db.bag)
        db.bag[i] = db.prior
    end
    return
end

"""
    getindex(db,i)

Retrieves the 1D Gaussian at index `i` from the distribution bag `db`
# Examples
```julia-repl
julia> db = DistributionBag(Gaussian1D())
Prior: μ = 0.0, σ = 1.0
  empty bag


julia> add!(db)
1

julia> add!(db)
2

julia> db
Prior: μ = 0.0, σ = 1.0
  [1]: μ = 0.0, σ = 1.0
  [2]: μ = 0.0, σ = 1.0


julia> db[1]
μ = 0.0, σ = 1.0
```
"""
function Base.getindex(d::DistributionBag, i::Int64)
    return (d.bag[i])
end

"""
Retrieves the 1D Gaussians at indices `indices` from the distribution bag `d`
# Examples 
```julia-repl
julia> db = DistributionBag(Gaussian1D())
Prior: μ = 0.0, σ = 1.0
  empty bag


julia> add!(db)
1

julia> add!(db)
2

julia> add!(db)
3

julia> db[[1,3]]
2-element Vector{Gaussian1D}:
 μ = 0.0, σ = 1.0
 μ = 0.0, σ = 1.0
 ```
"""
function Base.getindex(d::DistributionBag, indices::Vector{Int64})
    return d.bag[indices]
end

"""
    setindex(d,g,i)

Sets the 1D Gaussian at index `i` in the distribution bag `d` to `g`
# Examples
```julia-repl
julia> db = DistributionBag(Gaussian1D())
Prior: μ = 0.0, σ = 1.0
  empty bag


julia> add!(db)
1

julia> add!(db)
2

julia> db
Prior: μ = 0.0, σ = 1.0
  [1]: μ = 0.0, σ = 1.0
  [2]: μ = 0.0, σ = 1.0


julia> db[1] *= db[2]
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.setindex!(d::DistributionBag, g::Gaussian1D, i::Int64)
    d.bag[i] = g
end

"""
    firstindex(d)

Retrieves the index of the first element of the distribution bag
# Examples
```julia-repl
julia> db = DistributionBag(Gaussian1D())
Prior: μ = 0.0, σ = 1.0
  empty bag


julia> add!(db)
1

julia> add!(db)
2

julia> db
Prior: μ = 0.0, σ = 1.0
  [1]: μ = 0.0, σ = 1.0
  [2]: μ = 0.0, σ = 1.0


julia> db[1] *= db[2]
μ = 0.0, σ = 0.7071067811865476

julia> db[begin]
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.firstindex(d::DistributionBag)
    return (1)
end

"""
    lastindex(d)

Retrieves the index of the last element of the distribution bag
# Examples
```julia-repl
julia> db = DistributionBag(Gaussian1D())
Prior: μ = 0.0, σ = 1.0
  empty bag


julia> add!(db)
1

julia> add!(db)
2

julia> db
Prior: μ = 0.0, σ = 1.0
  [1]: μ = 0.0, σ = 1.0
  [2]: μ = 0.0, σ = 1.0


julia> db[2] *= db[1]
μ = 0.0, σ = 0.7071067811865476

julia> db[end]
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.lastindex(d::DistributionBag)
    return (length(d.bag))
end