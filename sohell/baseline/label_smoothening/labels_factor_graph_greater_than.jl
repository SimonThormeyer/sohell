include("factors.jl")
include("gaussian.jl")

function smoothen_labels_greater_than(noisy_labels::Vector{Float64}, β, ϵ)
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
    )

    previous_label_index = nothing
    # Build the factor graph
    for label ∈ noisy_labels
        label_index = add!(bag)
        push!(factors["result_indices"], label_index)
        label_prior = Gaussian1Dμσ2(label, β * β)
        label_factor = addFactor(GaussianFactor(label_prior, label_index, bag))
        push!(factors["BO_factors"], label_factor)

        if previous_label_index !== nothing
            difference_index = add!(bag)
            difference_factor = addFactor(WeightedSumFactor(1, -1, previous_label_index, label_index, difference_index, bag))
            push!(factors["Difference_factors"], difference_factor)

            greater_than_factor = addFactor(GreaterThanFactor(ϵ, difference_index, bag))
            push!(factors["Greater_than_factors"], greater_than_factor)
        end
        previous_label_index = label_index
    end

    # Pass priors into factors
    for bo_factor ∈ factors["BO_factors"]
        bo_factor.update!(1)
    end

    Δ = Inf
    it = 0
    while it < 1e3
        it += 1
        Δ = 0
        # Update difference and greater than factors, forward
        for (difference_factor, greater_than_factor) ∈ zip(factors["Difference_factors"], factors["Greater_than_factors"])
            Δ = max(Δ, difference_factor.update!(3))
            Δ = max(Δ, greater_than_factor.update!(1))
            Δ = max(Δ, difference_factor.update!(2))
        end

        # Update difference and greater than factors, backward
        for (difference_factor, greater_than_factor) ∈ Iterators.reverse(zip(factors["Difference_factors"], factors["Greater_than_factors"]))
            Δ = max(Δ, difference_factor.update!(3))
            Δ = max(Δ, greater_than_factor.update!(1))
            Δ = max(Δ, difference_factor.update!(1))
        end
        println("Δ: $Δ")
    end

    first(factors["Difference_factors"]).update!(1)
    last(factors["Difference_factors"]).update!(2)

    result = bag[factors["result_indices"]]

    return Dict("means" => [mean(gaussian) for gaussian in result], "stds" => [sqrt(variance(gaussian)) for gaussian in result], "Z" => exp(logNormalization(factorList, bag)))
end