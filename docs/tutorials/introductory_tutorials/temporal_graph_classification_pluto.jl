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
	using MLDatasets: TemporalBrains
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

The graph nodes represent brain regions and their number is fixed at 102 for each of the 27 snapshots, while the edges, representing functional connectivity, change over time.
For each snapshot, the feature of a node represents the average activation of the node during that snapshot.
Each temporal graph has a label representing gender ('M' for male and 'F' for female) and age group (22-25, 26-30, 31-35, and 36+).
The network's edge weights are binarized, and the threshold is set to 0.6 by default.
"

# ╔═╡ f2dbc66d-b8b7-46ae-ad5b-cbba1af86467
brain_dataset = TemporalBrains()

# ╔═╡ d9e4722d-6f02-4d41-955c-8bb3e411e404
md"After loading the dataset from the MLDatasets.jl package, we see that there are 1000 graphs and we need to convert them to the `TemporalSnapshotsGNNGraph` format.
So we create a function called `data_loader` that implements the latter and splits the dataset into the training set that will be used to train the model and the test set that will be used to test the performance of the model.
"

# ╔═╡ bb36237a-5545-47d0-a873-7ddff3efe8ba
function data_loader(graphs)
    dataset = Vector{TemporalSnapshotsGNNGraph}(undef, length(graphs))
    for i in 1:length(graphs)
        graph = graphs[i]
        dataset[i] = TemporalSnapshotsGNNGraph(GraphNeuralNetworks.mlgraph2gnngraph.(graph.snapshots))
		# Add graph and node features
        for t in 1:27
            dataset[i].snapshots[t].ndata.x = reduce(
                vcat, [I(102), dataset[i].snapshots[t].ndata.x'])
        end
        dataset[i].tgdata.g = Float32.(Array(Flux.onehot(graph.graph_data.g, ["F", "M"])))
    end
    # Split the dataset into a 80% training set and a 20% test set
    train_loader = dataset[1:800]
    test_loader = dataset[801:1000]
    return train_loader, test_loader
end

# ╔═╡ d4732340-9179-4ada-b82e-a04291d745c2
md"
The first part of the `data_loader` function calls the `mlgraph2gnngraph` function for each snapshot, which takes the graph and converts it into a `GNNGraph`. The vector of `GNNGraph`s is then rewritten to a `TemporalSnapshotsGNNGraph`.

The second part adds the graph and node feature to the temporal graphs, in particular adding the onehot encoding of the label of the graph and appending the mean activation of the node in the snapshot. For the graph frature, it adds the onehot encoding of the gender.

The last part splits the dataset.
"


# ╔═╡ ec088a59-2fc2-426a-a406-f8f8d6784128
md"
## Model

We now implement a simple model that takes a `TemporalSnapshotsGNNGraph` as input.
It consists of a `GINConv` applied independently to each snapshot, a `GlobalPool` to get an embedding for each snapshot, a pooling on the time dimension to get an embedding for the whole temporal graph, and finally a `Dense` layer.

First, we start by adapting the GlobalPool to the `TemporalSnapshotsGNNGraphs`.
"

# ╔═╡ 5ea98df9-4920-4c94-9472-3ef475af89fd
function (l::GlobalPool)(g::TemporalSnapshotsGNNGraph, x::AbstractVector)
    h = [reduce_nodes(l.aggr, g[i], x[i]) for i in 1:(g.num_snapshots)]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], length(h))
end

# ╔═╡ cfda2cf4-d08b-4f46-bd39-02ae3ed53369
md"
Then we implement the constructor of the model, which we call `GenderPredictionModel`, and the foward pass.
"

# ╔═╡ 2eedd408-67ee-47b2-be6f-2caec94e95b5
begin
	struct GenderPredictionModel
	    gin::GINConv
	    mlp::Chain
	    globalpool::GlobalPool
	    f::Function
	    dense::Dense
	end
	
	Flux.@functor GenderPredictionModel
	
	function GenderPredictionModel(; nfeatures = 103, nhidden = 128, activation = relu)
	    mlp = Chain(Dense(nfeatures, nhidden, activation), Dense(nhidden, nhidden, activation))
	    gin = GINConv(mlp, 0.5)
	    globalpool = GlobalPool(mean)
	    f = x -> mean(x, dims = 2)
	    dense = Dense(nhidden, 2)
	    GenderPredictionModel(gin, mlp, globalpool, f, dense)
	end
	
	function (m::GenderPredictionModel)(g::TemporalSnapshotsGNNGraph)
    h = m.gin(g, g.ndata.x)
    h = m.globalpool(g, h)
    h = m.f(h)
    m.dense(h)
	end
	
end

# ╔═╡ 76780020-406d-4803-9af0-d928e54fc18c
md"
## Training
"

# ╔═╡ d64be72e-8c1f-4551-b4f2-28c8b78466c0
function train(graphs; kws...)
	
    lossfunction(ŷ, y) = Flux.logitbinarycrossentropy(ŷ, y) 

    function eval_loss_accuracy(model, data_loader)
        error = mean([lossfunction(model(g), gpu(g.tgdata.g)) for g in data_loader])
        acc = mean([round(
                        100 *
                        mean(Flux.onecold(model(g)) .== Flux.onecold(gpu(g.tgdata.g)));
                        digits = 2) for g in data_loader])
        return (loss = error, acc = acc)
    end

    function report(epoch)
        train_loss, train_acc = eval_loss_accuracy(model, train_loader)
        test_loss, test_acc = eval_loss_accuracy(model, test_loader)
        println("Epoch: $epoch  $((; train_loss, train_acc))  $((; test_loss, test_acc))")
        return (train_loss, train_acc, test_loss, test_acc)
    end

    model = GenderPredictionModel() 

    opt = Flux.setup(Adam(1.0f-3), model)

    train_loader, test_loader = data_loader(graphs) # it takes a while to load the data

    report(0)
    for epoch in 1:200
        for g in train_loader
            grads = Flux.gradient(model) do model
                ŷ = model(g)
                lossfunction(vec(ŷ), g.tgdata.g)
            end
            Flux.update!(opt, model, grads[1])
        end
        if  epoch % 10 == 0
            report(epoch)
        end
    end
    return model
end


# ╔═╡ 483f17ba-871c-4769-88bd-8ec781d1909d
train(brain_dataset.graphs)

# ╔═╡ Cell order:
# ╟─69d00ec8-da47-11ee-1bba-13a14e8a6db2
# ╠═dfb02582-4dfa-4589-9dd5-c13bce0c44c3
# ╟─ef8406e4-117a-4cc6-9fa5-5028695b1a4f
# ╠═b8df1800-c69d-4e18-8a0a-097381b62a4c
# ╟─2544d468-1430-4986-88a9-be4df2a7cf27
# ╠═f2dbc66d-b8b7-46ae-ad5b-cbba1af86467
# ╠═d9e4722d-6f02-4d41-955c-8bb3e411e404
# ╠═bb36237a-5545-47d0-a873-7ddff3efe8ba
# ╟─d4732340-9179-4ada-b82e-a04291d745c2
# ╟─ec088a59-2fc2-426a-a406-f8f8d6784128
# ╠═5ea98df9-4920-4c94-9472-3ef475af89fd
# ╟─cfda2cf4-d08b-4f46-bd39-02ae3ed53369
# ╠═2eedd408-67ee-47b2-be6f-2caec94e95b5
# ╟─76780020-406d-4803-9af0-d928e54fc18c
# ╠═d64be72e-8c1f-4551-b4f2-28c8b78466c0
# ╠═483f17ba-871c-4769-88bd-8ec781d1909d
