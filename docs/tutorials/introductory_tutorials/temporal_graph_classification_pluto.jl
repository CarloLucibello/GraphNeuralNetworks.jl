### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ dfb02582-4dfa-4589-9dd5-c13bce0c44c3
begin
    using Pkg
    Pkg.develop("GraphNeuralNetworks")
    Pkg.add("MLDatasets")
    Pkg.add("Plots")
end

# ╔═╡ b8df1800-c69d-4e18-8a0a-097381b62a4c
begin
using Flux
using GraphNeuralNetworks
using Statistics, Random
using LinearAlgebra
using MLDatasets
end

# ╔═╡ 69d00ec8-da47-11ee-1bba-13a14e8a6db2
md"In this tutorial, we will learn how to extend the graph classification task to the case of temporal graphs, i.e., graphs whose topology and features are time-varying.

We will design and train a simple temporal graph neural network architecture to classify subjects' gender (female or male) using the temporal graphs extracted from their brain fMRI scan signals.
"

# ╔═╡ ef8406e4-117a-4cc6-9fa5-5028695b1a4f
md"
## Import

We start by importing the necessary libraries. We use `GraphNeuralNetworks.jl`, `Flux.jl` and `MLDatasets.jl`, among others.
"

# ╔═╡ 2544d468-1430-4986-88a9-be4df2a7cf27
md"
## Dataset: TemporalBrains
The TemporalBrains dataset contains a collection of functional brain connectivity networks from 1000 subjects obtained from resting-state functional MRI data from the [Human Connectome Project (HCP)](https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation). 
Functional connectivity is defined as the temporal dependence of neuronal activation patterns of anatomically separated brain regions.

The graph nodes represent brain regions and their number is fixed at 102 for each of the 27 snapshots, while the edges representing functional connectivity change over time.
For each snapshot, the feature of a node represents the average activation of the node during that snapshot.
Each temporal graph has a label representing gender ('M' for male and 'F' for female) and age group (22-25, 26-30, 31-35, and 36+).
The network's edge weights are binarized, and the threshold is set to 0.6 by default.
"

# ╔═╡ Cell order:
# ╟─69d00ec8-da47-11ee-1bba-13a14e8a6db2
# ╠═dfb02582-4dfa-4589-9dd5-c13bce0c44c3
# ╠═ef8406e4-117a-4cc6-9fa5-5028695b1a4f
# ╠═b8df1800-c69d-4e18-8a0a-097381b62a4c
# ╠═2544d468-1430-4986-88a9-be4df2a7cf27
