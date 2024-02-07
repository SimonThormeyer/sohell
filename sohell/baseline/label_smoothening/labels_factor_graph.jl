include("factors.jl")
include("gaussian.jl")

function smoothen_labels(noisy_labels::Vector{Float64}, τ::Float64, β::Float64, κ::Float64)
    bag = DistributionBag(Gaussian1D(0, 0))
    factorList = Vector{Factor}()

    # helper function to add factors to a long list
    function addFactor(f)
        push!(factorList, f)
        return (f)
    end


    factors = Dict(
        "BO_factors" => Factor[],
        "Dynamics_factors" => Factor[]
    )

    previous_dynamics_factor_index = nothing
    # Build the factor graph
    for label ∈ noisy_labels
        label_factor_index = add!(bag)
        label_prior = Gaussian1Dμσ2(label, label * label / (β * β))
        label_factor = addFactor(GaussianFactor(label_prior, label_factor_index, bag))
        push!(factors["BO_factors"], label_factor)

        if isnothing(previous_dynamics_factor_index)
            # For the first dymanics factor, take the BO factor Gaussian as a variable
            previous_dynamics_factor_index = label_factor_index
        end

        dynamics_factor = addFactor(GaussianMeanFactor(τ * τ, label_factor_index, previous_dynamics_factor_index, κ* τ, bag))
        push!(factors["Dynamics_factors"], dynamics_factor)
        previous_dynamics_factor_index = label_factor_index
    end

    # Pass priors into factors
    for bo_factor ∈ factors["BO_factors"]
        bo_factor.update!(1)
    end

    # Update dynamics factors, forward
    for dynamics_factor ∈ factors["Dynamics_factors"]
        dynamics_factor.update!(1)
    end

    for dynamics_factor ∈ Iterators.reverse(factors["Dynamics_factors"])
       dynamics_factor.update!(2)
    end

    result = bag.bag

    return Dict("means" => [mean(gaussian) for gaussian in result], "stds" => [sqrt(variance(gaussian)) for gaussian in result], "Z" => exp(logNormalization(factorList, bag)))
end

function smoothen_labels_measure_runtime(noisy_labels::Vector{Float64}, τ::Float64, β::Float64, κ::Float64)
    # Measure the runtime
    runtime = @elapsed result = smoothen_labels(noisy_labels, τ, β, κ)
    return merge(result, Dict("runtime" => runtime))
end