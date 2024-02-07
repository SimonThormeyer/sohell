include("labels_factor_graph_greater_than.jl")
include("labels_factor_graph.jl")
include("labels_factor_graph_combined.jl")
using PyCall
using Plots

py"""
import sys
sys.path.append("/Users/simon/projects/battery-analytics/src/learning/bayesian_regression")
sys.path.append("/Users/simon/projects/battery-analytics/src/simulator")

from labeling.cycle_finder import CycleData, CycleTuple, CycleFinder
from labeling.labeler import Labeler

cycle_finder = CycleFinder("../../cache")
cycles = cycle_finder.get_cycles_from_files_in_directory()[:-1]
simulation_parameters_list = [Labeler.load_parameters("../../cache/BO_doublet_new_ocv", cycle) for cycle in cycles]
labels = [parameters.get_average_sohc() for parameters in simulation_parameters_list]
"""

labels = py"labels"

# labels = labels[250:252]
# labels[3] -= 20
# anim = @animate for κ ∈ -2:0.5:2
    result_greater_than = smoothen_labels_greater_than(convert(Vector{Float64}, labels), 10, 0)
    result = smoothen_labels(convert(Vector{Float64}, labels), 1.0, 2.5, 0.0)
    # result_combined = smoothen_labels_combined(convert(Vector{Float64}, labels), 2.5, 0.5, 0.0)


    # println("Z = $(result["Z"])")
    # println("greather than Z = $(result_greater_than["Z"])")

    result = result_combined

    smoothened_labels_means = result["means"]
    smoothened_labels_stds = result["stds"]
    

    plot(smoothened_labels_means + smoothened_labels_stds * 1.96,
        linealpha=0,
        fillrange=smoothened_labels_means - smoothened_labels_stds * 1.96,
        fillalpha=0.35, c=1, label="95% confidence interval")

    scatter!([smoothened_labels_means labels], xlabel="Cycle Index", ylabel="SOH [%]", title="Label Smoothening", labels=["smoothened labels (mean)" "labels"])
# end
# gif(anim, "anim_fps15.gif", fps=15)

# histogram(labels - smoothened_labels_means, legend=false, title="Residuals")