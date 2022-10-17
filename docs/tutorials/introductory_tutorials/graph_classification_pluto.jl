### A Pluto.jl notebook ###
# v0.19.13

#> [frontmatter]
#> author = "[Carlo Lucibello](https://github.com/CarloLucibello)"
#> title = "Graph Classification with Graph Neural Networks"
#> date = "2022-05-23"
#> description = "Tutorial for Graph Classification using GraphNeuralNetworks.jl"
#> cover = "assets/graph_classification.gif"

using Markdown
using InteractiveUtils

# ╔═╡ c97a0002-2253-45b6-9266-017189dbb6fe
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(; temp=true)
    Pkg.add([
        PackageSpec(; name="GraphNeuralNetworks", version="0.4"),
        PackageSpec(; name="Flux", version="0.13"),
        PackageSpec(; name="MLDatasets", version="0.7"),
        PackageSpec(; name="MLUtils"),
	])
	Pkg.develop("GraphNeuralNetworks")
end

# ╔═╡ 361e0948-d91a-11ec-2d95-2db77435a0c1
# ╠═╡ show_logs = false
begin
	using Flux
	using Flux: onecold, onehotbatch, logitcrossentropy
	using Flux.Data: DataLoader
	using GraphNeuralNetworks
	using MLDatasets
	using MLUtils
	using LinearAlgebra, Random, Statistics
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
	Random.seed!(17) # for reproducibility
end;

# ╔═╡ 15136fd8-f9b2-4841-9a95-9de7b8969687
md"""
*This Pluto notebook is a julia adaptation of the Pytorch Geometric tutorials that can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html).*

In this tutorial session we will have a closer look at how to apply **Graph Neural Networks (GNNs) to the task of graph classification**.
Graph classification refers to the problem of classifying entire graphs (in contrast to nodes), given a **dataset of graphs**, based on some structural graph properties.
Here, we want to embed entire graphs, and we want to embed those graphs in such a way so that they are linearly separable given a task at hand.


The most common task for graph classification is **molecular property prediction**, in which molecules are represented as graphs, and the task may be to infer whether a molecule inhibits HIV virus replication or not.

The TU Dortmund University has collected a wide range of different graph classification datasets, known as the [**TUDatasets**](https://chrsmrrs.github.io/datasets/), which are also accessible via MLDatasets.jl.
Let's load and inspect one of the smaller ones, the **MUTAG dataset**:
"""

# ╔═╡ f6e86958-e96f-4c77-91fc-c72d8967575c
dataset = TUDataset("MUTAG")

# ╔═╡ 24f76360-8599-46c8-a49f-4c31f02eb7d8
dataset.graph_data.targets |> union

# ╔═╡ 5d5e5152-c860-4158-8bc7-67ee1022f9f8
g1, y1  = dataset[1] #get the first graph and target

# ╔═╡ 33163dd2-cb35-45c7-ae5b-d4854d141773
reduce(vcat, g.node_data.targets for (g,_) in dataset) |> union

# ╔═╡ a8d6a133-a828-4d51-83c4-fb44f9d5ede1
reduce(vcat, g.edge_data.targets for (g,_) in dataset)|> union

# ╔═╡ 3b3e0a79-264b-47d7-8bda-2a6db7290828
md"""
This dataset provides **188 different graphs**, and the task is to classify each graph into **one out of two classes**.

By inspecting the first graph object of the dataset, we can see that it comes with **17 nodes** and **38 edges**.
It also comes with exactly **one graph label**, and provides additional node labels (7 classes) and edge labels (4 classes).
However, for the sake of simplicity, we will not make use of edge labels.
"""

# ╔═╡ 7f7750ff-b7fa-4fe2-a5a8-6c9c26c479bb
md"""
We now convert the MLDatasets.jl graph types to our `GNNGraph`s and we also onehot encode both the node labels (which will be used as input features) and the graph labels (what we want to predict):  
"""

# ╔═╡ 936c09f6-ee62-4bc2-a0c6-749a66080fd2
begin
	graphs = mldataset2gnngraph(dataset)
	graphs = [GNNGraph(g, 
		               ndata=Float32.(onehotbatch(g.ndata.targets, 0:6)),
	                   edata=nothing) 
		      for g in graphs]
	y = onehotbatch(dataset.graph_data.targets, [-1, 1])
end

# ╔═╡ 2c6ccfdd-cf11-415b-b398-95e5b0b2bbd4
md"""We have some useful utilities for working with graph datasets, *e.g.*, we can shuffle the dataset and use the first 150 graphs as training graphs, while using the remaining ones for testing:
"""

# ╔═╡ 519477b2-8323-4ece-a7eb-141e9841117c
train_data, test_data = splitobs((graphs, y), at=150, shuffle=true) |> getobs

# ╔═╡ 3c3d5038-0ef6-47d7-a1b7-50880c5f3a0b
begin
	train_loader = DataLoader(train_data, batchsize=64, shuffle=true)
	test_loader = DataLoader(test_data, batchsize=64, shuffle=false)
end

# ╔═╡ f7778e2d-2e2a-4fc8-83b0-5242e4ec5eb4
md"""
Here, we opt for a `batch_size` of 64, leading to 3 (randomly shuffled) mini-batches, containing all ``2 \cdot 64+22 = 150`` graphs.
"""

# ╔═╡ 2a1c501e-811b-4ddd-887b-91e8c929c8b7
md"""
## Mini-batching of graphs

Since graphs in graph classification datasets are usually small, a good idea is to **batch the graphs** before inputting them into a Graph Neural Network to guarantee full GPU utilization.
In the image or language domain, this procedure is typically achieved by **rescaling** or **padding** each example into a set of equally-sized shapes, and examples are then grouped in an additional dimension.
The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the `batchsize`.

However, for GNNs the two approaches described above are either not feasible or may result in a lot of unnecessary memory consumption.
Therefore, GNN.jl opts for another approach to achieve parallelization across a number of examples. Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension (the last dimension).

This procedure has some crucial advantages over other batching procedures:

1. GNN operators that rely on a message passing scheme do not need to be modified since messages are not exchanged between two nodes that belong to different graphs.

2. There is no computational or memory overhead since adjacency matrices are saved in a sparse fashion holding only non-zero entries, *i.e.*, the edges.

GNN.jl can **batch multiple graphs into a single giant graph**:
"""


# ╔═╡ a142610a-d862-42a9-88af-c8d8b6825650
vec_gs, _ = first(train_loader)

# ╔═╡ 6faaf637-a0ff-468c-86b5-b0a7250258d6
MLUtils.batch(vec_gs)

# ╔═╡ e314b25f-e904-4c39-bf60-24cddf91fe9d
md"""
Each batched graph object is equipped with a **`graph_indicator` vector**, which maps each node to its respective graph in the batch:

```math
\textrm{graph-indicator} = [1, \ldots, 1, 2, \ldots, 2, 3, \ldots ]
```
"""

# ╔═╡ ac69571a-998b-4630-afd6-f3d405618bc5
md"""
## Training a Graph Neural Network (GNN)

Training a GNN for graph classification usually follows a simple recipe:

1. Embed each node by performing multiple rounds of message passing
2. Aggregate node embeddings into a unified graph embedding (**readout layer**)
3. Train a final classifier on the graph embedding

There exists multiple **readout layers** in literature, but the most common one is to simply take the average of node embeddings:

```math
\mathbf{x}_{\mathcal{G}} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathcal{x}^{(L)}_v
```

GNN.jl provides this functionality via `GlobalPool(mean)`, which takes in the node embeddings of all nodes in the mini-batch and the assignment vector `graph_indicator` to compute a graph embedding of size `[hidden_channels, batchsize]`.

The final architecture for applying GNNs to the task of graph classification then looks as follows and allows for complete end-to-end training:
"""


# ╔═╡ 04402032-18a4-42b5-ad04-19b286bd29b7
function create_model(nin, nh, nout)
	GNNChain(GCNConv(nin => nh, relu),
			 GCNConv(nh => nh, relu),
			 GCNConv(nh => nh),
			 GlobalPool(mean),
			 Dropout(0.5),
			 Dense(nh, nout))
end

# ╔═╡ 2313fd8d-6e84-4bde-bacc-fb697dc33cbb
md"""
Here, we again make use of the `GCNConv` with ``\mathrm{ReLU}(x) = \max(x, 0)`` activation for obtaining localized node embeddings, before we apply our final classifier on top of a graph readout layer.

Let's train our network for a few epochs to see how well it performs on the training as well as test set:
"""

# ╔═╡ c956ed97-fa5c-45c6-84dd-39f3e37d8070
function eval_loss_accuracy(model, data_loader, device)
    loss = 0.
    acc = 0.
    ntot = 0
    for (g, y) in data_loader
        g, y = MLUtils.batch(g) |> device, y |> device
        n = length(y)
        ŷ = model(g, g.ndata.x)
        loss += logitcrossentropy(ŷ, y) * n 
        acc += mean((ŷ .> 0) .== y) * n
        ntot += n
    end 
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end

# ╔═╡ 968c7087-7637-4844-9509-dd838cf99a8c
function train!(model; epochs=200, η=1e-2, infotime=10)
	# device = Flux.gpu # uncomment this for GPU training
	device = Flux.cpu
	model = model |> device
	ps = Flux.params(model)
    opt = Adam(1e-3)
	

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        @info (; epoch, train, test)
    end
    
    report(0)
    for epoch in 1:epochs
        for (g, y) in train_loader
            g, y = MLUtils.batch(g) |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
		epoch % infotime == 0 && report(epoch)
    end
end

# ╔═╡ dedf18d8-4281-49fa-adaf-bd57fc15095d
begin
	nin = 7  
	nh = 64
	nout = 2
	model = create_model(nin, nh, nout)
	train!(model)
end

# ╔═╡ 3454b311-9545-411d-b47a-b43724b84c36
md"""
As one can see, our model reaches around **74% test accuracy**.
Reasons for the fluctuations in accuracy can be explained by the rather small dataset (only 38 test graphs), and usually disappear once one applies GNNs to larger datasets.

## (Optional) Exercise

Can we do better than this?
As multiple papers pointed out ([Xu et al. (2018)](https://arxiv.org/abs/1810.00826), [Morris et al. (2018)](https://arxiv.org/abs/1810.02244)), applying **neighborhood normalization decreases the expressivity of GNNs in distinguishing certain graph structures**.
An alternative formulation ([Morris et al. (2018)](https://arxiv.org/abs/1810.02244)) omits neighborhood normalization completely and adds a simple skip-connection to the GNN layer in order to preserve central node information:

```math
\mathbf{x}_i^{(\ell+1)} = \mathbf{W}^{(\ell + 1)}_1 \mathbf{x}_i^{(\ell)} + \mathbf{W}^{(\ell + 1)}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j^{(\ell)}
```

This layer is implemented under the name `GraphConv` in GNN.jl.

As an exercise, you are invited to complete the following code to the extent that it makes use of `GraphConv` rather than `GCNConv`.
This should bring you close to **82% test accuracy**.
"""



# ╔═╡ 93e08871-2929-4279-9f8a-587168617365
md"""
## Conclusion

In this chapter, you have learned how to apply GNNs to the task of graph classification.
You have learned how graphs can be batched together for better GPU utilization, and how to apply readout layers for obtaining graph embeddings rather than node embeddings.
"""

# ╔═╡ Cell order:
# ╟─c97a0002-2253-45b6-9266-017189dbb6fe
# ╟─361e0948-d91a-11ec-2d95-2db77435a0c1
# ╟─15136fd8-f9b2-4841-9a95-9de7b8969687
# ╠═f6e86958-e96f-4c77-91fc-c72d8967575c
# ╠═24f76360-8599-46c8-a49f-4c31f02eb7d8
# ╠═5d5e5152-c860-4158-8bc7-67ee1022f9f8
# ╠═33163dd2-cb35-45c7-ae5b-d4854d141773
# ╠═a8d6a133-a828-4d51-83c4-fb44f9d5ede1
# ╟─3b3e0a79-264b-47d7-8bda-2a6db7290828
# ╟─7f7750ff-b7fa-4fe2-a5a8-6c9c26c479bb
# ╠═936c09f6-ee62-4bc2-a0c6-749a66080fd2
# ╟─2c6ccfdd-cf11-415b-b398-95e5b0b2bbd4
# ╠═519477b2-8323-4ece-a7eb-141e9841117c
# ╠═3c3d5038-0ef6-47d7-a1b7-50880c5f3a0b
# ╟─f7778e2d-2e2a-4fc8-83b0-5242e4ec5eb4
# ╟─2a1c501e-811b-4ddd-887b-91e8c929c8b7
# ╠═a142610a-d862-42a9-88af-c8d8b6825650
# ╠═6faaf637-a0ff-468c-86b5-b0a7250258d6
# ╟─e314b25f-e904-4c39-bf60-24cddf91fe9d
# ╟─ac69571a-998b-4630-afd6-f3d405618bc5
# ╠═04402032-18a4-42b5-ad04-19b286bd29b7
# ╟─2313fd8d-6e84-4bde-bacc-fb697dc33cbb
# ╠═c956ed97-fa5c-45c6-84dd-39f3e37d8070
# ╠═968c7087-7637-4844-9509-dd838cf99a8c
# ╠═dedf18d8-4281-49fa-adaf-bd57fc15095d
# ╟─3454b311-9545-411d-b47a-b43724b84c36
# ╟─93e08871-2929-4279-9f8a-587168617365
