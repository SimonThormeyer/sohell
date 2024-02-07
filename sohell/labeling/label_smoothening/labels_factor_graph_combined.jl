include("factors.jl")
include("gaussian.jl")

function smoothen_labels_combined(noisy_labels::Vector{Float64}, β, τ, ϵ)
    bag = DistributionBag(Gaussian1D(0, 0))
    factorList = Vector{Factor}()

    # helper function to add factors to a long list
    function addFactor(f::Factor)
        push!(factorList, f)
        return (f)
    end


    factors = Dict(
        "BO_factors" => Factor[],
        "result_indices" => Int64[],
        "Difference_factors" => Factor[],
        "Greater_than_factors" => Factor[],
        "Mean_factors" => Factor[],
    )

    previous_latent_soh_index = nothing
    # Build the factor graph
    for measured_soh ∈ noisy_labels
        measured_soh_index = add!(bag)
        push!(factors["result_indices"], measured_soh_index)

        measured_soh_prior = Gaussian1Dμσ2(measured_soh, β * β)
        measured_soh_factor = addFactor(GaussianFactor(measured_soh_prior, measured_soh_index, bag))
        push!(factors["BO_factors"], measured_soh_factor)

        latent_soh_index = add!(bag)
        mean_factor = addFactor(GaussianMeanFactor(τ * τ, latent_soh_index, measured_soh_index, bag))
        push!(factors["Mean_factors"], mean_factor)

        if previous_latent_soh_index !== nothing
            difference_index = add!(bag)
            difference_factor = addFactor(WeightedSumFactor(1, -1, previous_latent_soh_index, latent_soh_index, difference_index, bag))
            push!(factors["Difference_factors"], difference_factor)

            greater_than_factor = addFactor(GreaterThanFactor(ϵ, difference_index, bag))
            push!(factors["Greater_than_factors"], greater_than_factor)
        end
        previous_latent_soh_index = latent_soh_index
    end

    # Pass priors into factors
    for bo_factor ∈ factors["BO_factors"]
        bo_factor.update!(1)
    end

    # Update mean factors, forward
    for mean_factor ∈ factors["Mean_factors"]
        mean_factor.update!(1)
    end

    Δ = Inf
    it = 0
    # while Δ > 2e-6
    while it < 1e4
        it += 1
        Δ = 0
        # Update greater than factors, forward
        for (difference_factor, greater_than_factor) ∈ zip(factors["Difference_factors"], factors["Greater_than_factors"], factors["Mean_factors"], factors["Mean_factors"][2:lastindex(factors["Mean_factors"])])
            Δ = max(Δ, difference_factor.update!(3))
            Δ = max(Δ, greater_than_factor.update!(1))
            Δ = max(Δ, difference_factor.update!(2))
        end

        # Update greater than factors, backward
        for (difference_factor, greater_than_factor) ∈ Iterators.reverse(zip(factors["Difference_factors"], factors["Greater_than_factors"]))
            Δ = max(Δ, difference_factor.update!(3))
            Δ = max(Δ, greater_than_factor.update!(1))
            Δ = max(Δ, difference_factor.update!(1))
        end

        println("Δ: $Δ")
    end

    first(factors["Difference_factors"]).update!(1)
    last(factors["Difference_factors"]).update!(2)

    # Update mean factors, backward
    for mean_factor ∈ Iterators.reverse(factors["Mean_factors"])
        mean_factor.update!(2)
    end

    result = bag[factors["result_indices"]]

    return Dict("means" => [mean(gaussian) for gaussian in result], "stds" => [sqrt(variance(gaussian)) for gaussian in result], "Z" => exp(logNormalization(factorList, bag)))
end