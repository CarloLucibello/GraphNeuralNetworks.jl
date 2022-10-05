### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 6f20e59c-b002-4d22-9ee0-b62596574776
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(; temp=true)
    packages = [
        PackageSpec(; name="GraphNeuralNetworks", version="0.4"),
        PackageSpec(; name="Flux", version="0.13"),
        PackageSpec(; name="MLDatasets", version="0.7"),
        PackageSpec(; name="GraphMakie"),
        PackageSpec(; name="Graphs"),
        PackageSpec(; name="CairoMakie"),
        PackageSpec(; name="PlutoUI"),
    ]
    Pkg.add(packages)
end

# ╔═╡ 361e0948-d91a-11ec-2d95-2db77435a0c1
begin
	using Flux
	using Flux: onecold, onehotbatch, logitcrossentropy
	using GraphNeuralNetworks
	import MLDatasets
	using LinearAlgebra, Random, Statistics
	import GraphMakie
	import CairoMakie as Makie
	using Graphs
	using PlutoUI
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # don't ask for dataset download confirmation
	Random.seed!(17) # for reproducibility
end;

# ╔═╡ cc051aa1-b929-4bca-b261-7f797a644a2b
md"""
---
title: Hands-on introduction to Graph Neural Networks
cover: assets/logo.svg
author: "[Carlo Lucibello](https://github.com/CarloLucibello)"
date: 2022-05-24
description: A beginner level introduction to graph machine learning using GraphNeuralNetworks.jl.
---
"""

# ╔═╡ 03a9e023-e682-4ea3-a10b-14c4d101b291
md"""
*This Pluto notebook is a julia adaptation of the Pytorch Geometric tutorials that can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html).*

Recently, deep learning on graphs has emerged to one of the hottest research fields in the deep learning community.
Here, **Graph Neural Networks (GNNs)** aim to generalize classical deep learning concepts to irregular structured data (in contrast to images or texts) and to enable neural networks to reason about objects and their relations.

This is done by following a simple **neural message passing scheme**, where node features ``\mathbf{x}_i^{(\ell)}`` of all nodes ``i \in \mathcal{V}`` in a graph ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` are iteratively updated by aggregating localized information from their neighbors ``\mathcal{N}(i)``:

```math
\mathbf{x}_i^{(\ell + 1)} = f^{(\ell + 1)}_{\theta} \left( \mathbf{x}_i^{(\ell)}, \left\{ \mathbf{x}_j^{(\ell)} : j \in \mathcal{N}(i) \right\} \right)
```

This tutorial will introduce you to some fundamental concepts regarding deep learning on graphs via Graph Neural Networks based on the **[GraphNeuralNetworks.jl library](https://github.com/CarloLucibello/GraphNeuralNetworks.jl)**.
GNN.jl is an extension library to the popular deep learning framework [Flux.jl](https://fluxml.ai/Flux.jl/stable/), and consists of various methods and utilities to ease the implementation of Graph Neural Networks.

Let's first import the packages we need:
"""

# ╔═╡ ef96f5ae-724d-4b8e-b7d7-c116ad1c3279
md"""
Following [Kipf et al. (2017)](https://arxiv.org/abs/1609.02907), let's dive into the world of GNNs by looking at a simple graph-structured example, the well-known [**Zachary's karate club network**](https://en.wikipedia.org/wiki/Zachary%27s_karate_club). This graph describes a social network of 34 members of a karate club and documents links between members who interacted outside the club. Here, we are interested in detecting communities that arise from the member's interaction.

GNN.jl provides utilities to convert [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl)'s datasets to its own type:
"""

# ╔═╡ 4ba372d4-7a6a-41e0-92a0-9547a78e2898
dataset = MLDatasets.KarateClub()

# ╔═╡ 55aca2f0-4bbb-4d3a-9777-703896cfc548
md"""
After initializing the `KarateClub` dataset, we first can inspect some of its properties.
For example, we can see that this dataset holds exactly **one graph**.
Furthermore, the graph holds exactly **4 classes**, which represent the community each node belongs to.
"""

# ╔═╡ a1d35896-0f52-4c8b-b7dc-ec65649237c8
karate = dataset[1]

# ╔═╡ 48d7df25-9190-45c9-9829-140f452e5151
karate.node_data.labels_comm

# ╔═╡ 4598bf67-5448-4ce5-8be8-a473ab1a6a07
md"""
Now we convert the single-graph dataset to a `GNNGraph`. Moreover, we add a an array of node features, a **34-dimensional feature vector**  for each node which uniquely describes the members of the karate club. We also add a training mask selecting the nodes to be used for training in our semi-supervised node classification task.
"""

# ╔═╡ 8d41a9fa-eefe-40c9-8cc3-cd503cf7434d
begin 
	# convert a MLDataset.jl's dataset to a GNNGraphs (or a collection of graphs)
	g = mldataset2gnngraph(dataset)
	
	x = zeros(Float32, g.num_nodes, g.num_nodes)
	x[diagind(x)] .= 1
	
	train_mask = [ true, false, false, false,  true, false, false, false,  true,
		false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false,  true, false, false, false, false, false,
        false, false, false, false]

	labels = g.ndata.labels_comm
	y = onehotbatch(labels, 0:3)
	
	g = GNNGraph(g, ndata=(; x, y, train_mask))
end

# ╔═╡ c42c7f73-f84e-4e72-9af4-a6421af57f0d
md"""
Let's now look at the underlying graph in more detail:
"""

# ╔═╡ a7ad9de3-3e18-4aff-b118-a4d798a2f4ec
with_terminal() do
	# Gather some statistics about the graph.
	println("Number of nodes: $(g.num_nodes)")
	println("Number of edges: $(g.num_edges)")
	println("Average node degree: $(g.num_edges / g.num_nodes)")
	println("Number of training nodes: $(sum(g.ndata.train_mask))")
	println("Training node label rate: $(mean(g.ndata.train_mask))")
	# println("Has isolated nodes: $(has_isolated_nodes(g))")
	println("Has self-loops: $(has_self_loops(g))")
	println("Is undirected: $(is_bidirected(g))")
end

# ╔═╡ 1e362709-a0d0-45d5-b2fd-a91c45fa317a
md"""
Each graph in GNN.jl is represented by a  `GNNGraph` object, which holds all the information to describe its graph representation.
We can print the data object anytime via `print(g)` to receive a short summary about its attributes and their shapes.

The  `g` object holds 3 attributes:
- `g.ndata` contains node related information;
- `g.edata` holds edge-related information;
- `g.gdata`: this stores the global data, therefore neither node nor edge specific features.

These attributes are `NamedTuples` that can store multiple feature arrays: we can access a specific set of features e.g. `x`, with `g.ndata.x`.


In our task, `g.ndata.train_mask` describes for which nodes we already know their community assignments. In total, we are only aware of the ground-truth labels of 4 nodes (one for each community), and the task is to infer the community assignment for the remaining nodes.

The `g` object also provides some **utility functions** to infer some basic properties of the underlying graph.
For example, we can easily infer whether there exists isolated nodes in the graph (*i.e.* there exists no edge to any node), whether the graph contains self-loops (*i.e.*, ``(v, v) \in \mathcal{E}``), or whether the graph is bidirected (*i.e.*, for each edge ``(v, w) \in \mathcal{E}`` there also exists the edge ``(w, v) \in \mathcal{E}``).

Let us now inspect the `edge_index` method:

"""

# ╔═╡ d627736a-fd5a-4cdc-bd4e-89ff8b8c55bd
edge_index(g)

# ╔═╡ 98bb86d2-a7b9-4110-8851-8829a9f9b4d0
md"""
By printing `edge_index(g)`, we can understand how GNN.jl represents graph connectivity internally.
We can see that for each edge, `edge_index` holds a tuple of two node indices, where the first value describes the node index of the source node and the second value describes the node index of the destination node of an edge.

This representation is known as the **COO format (coordinate format)** commonly used for representing sparse matrices.
Instead of holding the adjacency information in a dense representation ``\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}``, GNN.jl represents graphs sparsely, which refers to only holding the coordinates/values for which entries in ``\mathbf{A}`` are non-zero.

Importantly, GNN.jl does not distinguish between directed and undirected graphs, and treats undirected graphs as a special case of directed graphs in which reverse edges exist for every entry in the edge_index.

Since a `GNNGraph` is an `AbstractGraph` from the `Graphs.jl` library, it supports graph algorithms and visualization tools from the wider julia graph ecosystem:
"""

# ╔═╡ 9820cc77-ae0a-454a-86b6-a23dbc56b6fd
GraphMakie.graphplot(g |> to_unidirected, node_size=20, node_color=labels, arrow_show=false) 

# ╔═╡ 86135c51-950c-4c08-b9e0-6c892234ff87
md"""

## Implementing Graph Neural Networks

After learning about GNN.jl's data handling, it's time to implement our first Graph Neural Network!

For this, we will use on of the most simple GNN operators, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)), which is defined as

```math
\mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
```

where ``\mathbf{W}^{(\ell + 1)}`` denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and ``c_{w,v}`` refers to a fixed normalization coefficient for each edge.

GNN.jl implements this layer via `GCNConv`, which can be executed by passing in the node feature representation `x` and the COO graph connectivity representation `edge_index`.

With this, we are ready to create our first Graph Neural Network by defining our network architecture:
"""

# ╔═╡ 88d1e59f-73d6-46ee-87e8-35beb7bc7674
begin 
	struct GCN
		layers::NamedTuple
	end
	
	Flux.@functor GCN # provides parameter collection, gpu movement and more

	function GCN(num_features, num_classes)
		layers = (conv1 = GCNConv(num_features => 4),
		          conv2 = GCNConv(4 => 4),
		          conv3 = GCNConv(4 => 2),
		          classifier = Dense(2, num_classes))
		return GCN(layers)
	end

	function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
	    l = gcn.layers
		x = l.conv1(g, x)
        x = tanh.(x)
        x = l.conv2(g, x)
        x = tanh.(x)
        x = l.conv3(g, x)
        x = tanh.(x)  # Final GNN embedding space.
        out = l.classifier(x)
        # Apply a final (linear) classifier.
        return out, x
	end
end

# ╔═╡ 9838189c-5cf6-4f21-b58e-3bb905408ad3
md"""

Here, we first initialize all of our building blocks in the constructor and define the computation flow of our network in the call method.
We first define and stack **three graph convolution layers**, which corresponds to aggregating 3-hop neighborhood information around each node (all nodes up to 3 "hops" away).
In addition, the `GCNConv` layers reduce the node feature dimensionality to ``2``, *i.e.*, ``34 \rightarrow 4 \rightarrow 4 \rightarrow 2``. Each `GCNConv` layer is enhanced by a `tanh` non-linearity.

After that, we apply a single linear transformation (`Flux.Dense` that acts as a classifier to map our nodes to 1 out of the 4 classes/communities.

We return both the output of the final classifier as well as the final node embeddings produced by our GNN.
We proceed to initialize our final model via `GCN()`, and printing our model produces a summary of all its used sub-modules.

### Embedding the Karate Club Network

Let's take a look at the node embeddings produced by our GNN.
Here, we pass in the initial node features `x` and the graph  information `g` to the model, and visualize its 2-dimensional embedding.
"""


# ╔═╡ ad2c2e51-08ec-4ddc-9b5c-668a3688db12
begin 
	num_features = 34
	num_classes = 4
	gcn = GCN(num_features, num_classes)
end

# ╔═╡ ce26c963-0438-4ab2-b5c6-520272beef2b
_, h = gcn(g, g.ndata.x)

# ╔═╡ e545e74f-0a3c-4d18-9cc7-557ca60be567
function visualize_embeddings(h; colors=nothing)
	xs = h[1,:] |> vec
	ys = h[2,:] |> vec
	Makie.scatter(xs, ys, color=labels, markersize= 20)
end

# ╔═╡ 26138606-2e8d-435b-aa1a-b6159a0d2739
visualize_embeddings(h, colors=labels)

# ╔═╡ b9359c7d-b7fe-412d-8f5e-55ba6bccb4e9
md"""
Remarkably, even before training the weights of our model, the model produces an embedding of nodes that closely resembles the community-structure of the graph.
Nodes of the same color (community) are already closely clustered together in the embedding space, although the weights of our model are initialized **completely at random** and we have not yet performed any training so far!
This leads to the conclusion that GNNs introduce a strong inductive bias, leading to similar embeddings for nodes that are close to each other in the input graph.

### Training on the Karate Club Network

But can we do better? Let's look at an example on how to train our network parameters based on the knowledge of the community assignments of 4 nodes in the graph (one for each community):

Since everything in our model is differentiable and parameterized, we can add some labels, train the model and observe how the embeddings react.
Here, we make use of a semi-supervised or transductive learning procedure: We simply train against one node per class, but are allowed to make use of the complete input graph data.

Training our model is very similar to any other Flux model.
In addition to defining our network architecture, we define a loss criterion (here, `logitcrossentropy` and initialize a stochastic gradient optimizer (here, `Adam`).
After that, we perform multiple rounds of optimization, where each round consists of a forward and backward pass to compute the gradients of our model parameters w.r.t. to the loss derived from the forward pass.
If you are not new to Flux, this scheme should appear familiar to you. 

Note that our semi-supervised learning scenario is achieved by the following line:
```
loss = logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
```
While we compute node embeddings for all of our nodes, we **only make use of the training nodes for computing the loss**.
Here, this is implemented by filtering the output of the classifier `out` and ground-truth labels `data.y` to only contain the nodes in the `train_mask`.

Let us now start training and see how our node embeddings evolve over time (best experienced by explicitly running the code):
"""


# ╔═╡ 912560a1-9c72-47bd-9fce-9702b346b603
begin
	model = GCN(num_features, num_classes)
    ps = Flux.params(model)
    opt = Adam(1e-2)
	epochs = 2000

	emb = h
	function report(epoch, loss, h)
		# p = visualize_embeddings(h)
		@info (; epoch, loss)
	end
	
	report(0, 10., emb)
	for epoch in 1:epochs
        loss, gs = Flux.withgradient(ps) do
			ŷ, emb = model(g, g.ndata.x)
            logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
        end
		
        Flux.Optimise.update!(opt, ps, gs)
		if epoch % 200 == 0
			report(epoch, loss, emb)
		end
	end
end

# ╔═╡ c8a217c9-0087-41f0-90c8-aac29bc1c996
ŷ, emb_final = model(g, g.ndata.x)

# ╔═╡ 727b24bc-0b1e-4ebd-b8ef-987015751e38
# train accuracy
mean(onecold(ŷ[:, train_mask]) .== onecold(y[:, train_mask]))

# ╔═╡ 8c60ec7e-46b0-40f7-bf6a-6228a31e1f66
# test accuracy
mean(onecold(ŷ[:, .!train_mask]) .== onecold(y[:, .!train_mask]))

# ╔═╡ 44d9f8cf-1023-48ad-a01f-07e59f4b4226
visualize_embeddings(emb_final, colors=labels)

# ╔═╡ a8841d35-97f9-431d-acab-abf478ce91a9
md"""
As one can see, our 3-layer GCN model manages to linearly separating the communities and classifying most of the nodes correctly.

Furthermore, we did this all with a few lines of code, thanks to the GraphNeuralNetworks.jl which helped us out with data handling and GNN implementations.
"""

# ╔═╡ Cell order:
# ╟─cc051aa1-b929-4bca-b261-7f797a644a2b
# ╟─03a9e023-e682-4ea3-a10b-14c4d101b291
# ╟─6f20e59c-b002-4d22-9ee0-b62596574776
# ╠═361e0948-d91a-11ec-2d95-2db77435a0c1
# ╟─ef96f5ae-724d-4b8e-b7d7-c116ad1c3279
# ╠═4ba372d4-7a6a-41e0-92a0-9547a78e2898
# ╟─55aca2f0-4bbb-4d3a-9777-703896cfc548
# ╠═a1d35896-0f52-4c8b-b7dc-ec65649237c8
# ╠═48d7df25-9190-45c9-9829-140f452e5151
# ╟─4598bf67-5448-4ce5-8be8-a473ab1a6a07
# ╠═8d41a9fa-eefe-40c9-8cc3-cd503cf7434d
# ╟─c42c7f73-f84e-4e72-9af4-a6421af57f0d
# ╠═a7ad9de3-3e18-4aff-b118-a4d798a2f4ec
# ╟─1e362709-a0d0-45d5-b2fd-a91c45fa317a
# ╠═d627736a-fd5a-4cdc-bd4e-89ff8b8c55bd
# ╟─98bb86d2-a7b9-4110-8851-8829a9f9b4d0
# ╠═9820cc77-ae0a-454a-86b6-a23dbc56b6fd
# ╟─86135c51-950c-4c08-b9e0-6c892234ff87
# ╠═88d1e59f-73d6-46ee-87e8-35beb7bc7674
# ╟─9838189c-5cf6-4f21-b58e-3bb905408ad3
# ╠═ad2c2e51-08ec-4ddc-9b5c-668a3688db12
# ╠═ce26c963-0438-4ab2-b5c6-520272beef2b
# ╠═e545e74f-0a3c-4d18-9cc7-557ca60be567
# ╠═26138606-2e8d-435b-aa1a-b6159a0d2739
# ╟─b9359c7d-b7fe-412d-8f5e-55ba6bccb4e9
# ╠═912560a1-9c72-47bd-9fce-9702b346b603
# ╠═c8a217c9-0087-41f0-90c8-aac29bc1c996
# ╠═727b24bc-0b1e-4ebd-b8ef-987015751e38
# ╠═8c60ec7e-46b0-40f7-bf6a-6228a31e1f66
# ╠═44d9f8cf-1023-48ad-a01f-07e59f4b4226
# ╟─a8841d35-97f9-431d-acab-abf478ce91a9
