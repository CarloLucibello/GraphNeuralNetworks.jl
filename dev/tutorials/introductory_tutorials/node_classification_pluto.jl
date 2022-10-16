### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 2c710e0f-4275-4440-a3a9-27eabf61823a
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(; temp=true)
    packages = [
        PackageSpec(; name="GraphNeuralNetworks", version="0.4"),
        PackageSpec(; name="Flux", version="0.13"),
        PackageSpec(; name="MLDatasets", version="0.7"),
        PackageSpec(; name="Plots"),
        PackageSpec(; name="TSne"),
        PackageSpec(; name="PlutoUI"),
    ]
    Pkg.add(packages)
end

# ╔═╡ 5463330a-0161-11ed-1b18-936030a32bbf
# ╠═╡ show_logs = false
begin
    using MLDatasets
    using GraphNeuralNetworks
    using Flux
    using Flux: onecold, onehotbatch, logitcrossentropy
    using Plots
    using PlutoUI
    using TSne
    using Random
    using Statistics
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    Random.seed!(17) # for reproducibility
end

# ╔═╡ 8db76e69-01ee-42d6-8721-19a3848693ae
md"""
---
title: Node Classification with Graph Neural Networks
cover: assets/node_classsification.gif
author: "[Deeptendu Santra](https://github.com/Dsantra92)"
date: 2022-09-25
description: Tutorial for Node classification using GraphNeuralNetworks.jl
---
"""

# ╔═╡ ca2f0293-7eac-4d9a-9a2f-fda47fd95a99
md"""
Following our previous tutorial in GNNs, we covered how to create graph neural networks.

In this tutorial, we will be learning how to use Graph Neural Networks (GNNs) for node classification. Given the ground-truth labels of only a small subset of nodes, and want to infer the labels for all the remaining nodes (transductive learning).
"""

# ╔═╡ 4455f18c-2bd9-42ed-bce3-cfe6561eab23
md"""
## Import
Let us start off by importing some libraries. We will be using Flux.jl and `GraphNeuralNetworks.jl` for our tutorial.
"""

# ╔═╡ 0d556a7c-d4b6-4cef-806c-3e1712de0791
md"""
## Visualize
We want to visualize the the outputs of the resutls using t-distributed stochastic neighbor embedding (tsne) to embed our output embeddings onto a 2D plane.
"""

# ╔═╡ 997b5387-3811-4998-a9d1-7981b58b9e09
function visualize_tsne(out, targets)
    z = tsne(out, 2)
    scatter(z[:, 1], z[:, 2], color=Int.(targets[1:size(z,1)]), leg = false)
end

# ╔═╡ 4b6fa18d-7ccd-4c07-8dc3-ded4d7da8562
md"""
## Dataset: Cora

For our tutorial, we will be using the `Cora` dataset. `Cora` is a citaton network of 2708 documents classified into one of seven classes and 5429 links. Each node represent articles/documents and the edges between these nodes if one of them cite each other.

Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.

This dataset was first introduced by [Yang et al. (2016)](https://arxiv.org/abs/1603.08861) as one of the datasets of the `Planetoid` benchmark suite. We will be using [MLDatasets.jl](https://juliaml.github.io/MLDatasets.jl/stable/) for an easy accss to this dataset.
"""

# ╔═╡ edab1e3a-31f6-471f-9835-5b1f97e5cf3f
dataset = Cora()

# ╔═╡ d73a2db5-9417-4b2c-a9f5-b7d499a53fcb
md"""
Datasets in MLDatasets.jl have `metadata` containing information about the dataset itself.
"""

# ╔═╡ 32bb90c1-c802-4c0c-a620-5d3b8f3f2477
dataset.metadata

# ╔═╡ 3438ee7f-bfca-465d-85df-13379622d415
md"""
The `graphs` variable GraphDataset contains the graph. The `Cora` dataaset contains only 1 graph.
"""

# ╔═╡ eec6fb60-0774-4f2a-bcb7-dbc28ab747a6
dataset.graphs

# ╔═╡ bd2fd04d-7fb0-4b31-959b-bddabe681754
md"""
There is only one graph of the dataset. The `node_data` contians `features` indicating if certain words are present or not and `targets` indicating the class for each document. We convert the single-graph dataset to a `GNNGraph`.
"""

# ╔═╡ b29c3a02-c21b-4b10-aa04-b90bcc2931d8
g = mldataset2gnngraph(dataset)

# ╔═╡ 16d9fbad-d4dc-4b51-9576-1736d228e2b3
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

# ╔═╡ 923d061c-25c3-4826-8147-9afa3dbd5bac
md"""
Overall, this dataset is quite similar to the previously used [`KarateClub`](https://juliaml.github.io/MLDatasets.jl/stable/datasets/graphs/#MLDatasets.KarateClub) network.
We can see that the `Cora` network holds 2,708 nodes and 10,556 edges, resulting in an average node degree of 3.9.
For training this dataset, we are given the ground-truth categories of 140 nodes (20 for each class).
This results in a training node label rate of only 5%.

We can further see that this network is undirected, and that there exists no isolated nodes (each document has at least one citation).
"""

# ╔═╡ 28e00b95-56db-4d36-a205-fd24d3c54e17
begin
    x = g.ndata.features
    # we onehot encode both the node labels (what we want to predict):
    y = onehotbatch(g.ndata.targets, 1:7)
    train_mask = g.ndata.train_mask
    num_features = size(x)[1]
    hidden_channels = 16
    num_classes = dataset.metadata["num_classes"]
end

# ╔═╡ fa743000-604f-4d28-99f1-46ab2f884b8e
md"""
## Multi-layer Perception Network (MLP)

In theory, we should be able to infer the category of a document solely based on its content, *i.e.* its bag-of-words feature representation, without taking any relational information into account.

Let's verify that by constructing a simple MLP that solely operates on input node features (using shared weights across all nodes):
"""

# ╔═╡ f972f61b-2001-409b-9190-ac2c0652829a
begin
    struct MLP
        layers::NamedTuple
    end

    Flux.@functor MLP
    
    function MLP(num_features, num_classes, hidden_channels; drop_rate=0.5)
        layers = (hidden = Dense(num_features => hidden_channels),
                    drop = Dropout(drop_rate),
                    classifier = Dense(hidden_channels => num_classes))
        return MLP(layers)
    end

    function (model::MLP)(x::AbstractMatrix)
        l = model.layers
        x = l.hidden(x)
        x = relu(x)
        x = l.drop(x)
        x = l.classifier(x)
        return x
    end
end

# ╔═╡ 4dade64a-e28e-42c7-8ad5-93fc04724d4d
md"""
### Training a Multilayer Perceptron

Our MLP is defined by two linear layers and enhanced by [ReLU](https://fluxml.ai/Flux.jl/stable/models/nnlib/#NNlib.relu) non-linearity and [Dropout](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout).
Here, we first reduce the 1433-dimensional feature vector to a low-dimensional embedding (`hidden_channels=16`), while the second linear layer acts as a classifier that should map each low-dimensional node embedding to one of the 7 classes.

Let's train our simple MLP by following a similar procedure as described in [the first part of this tutorial](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/tutorials/gnn_intro_pluto).
We again make use of the **cross entropy loss** and **Adam optimizer**.
This time, we also define a **`accuracy` function** to evaluate how well our final model performs on the test node set (which labels have not been observed during training).
"""

# ╔═╡ 05979cfe-439c-4abc-90cd-6ca2a05f6e0f
function train(model::MLP, data::AbstractMatrix, epochs::Int, opt, ps)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, gs = Flux.withgradient(ps) do
            ŷ = model(data)
            logitcrossentropy(ŷ[:, train_mask], y[:, train_mask])
        end
    
        Flux.Optimise.update!(opt, ps, gs)
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end

# ╔═╡ a3f420e1-7521-4df9-b6d5-fc0a1fd05095
function accuracy(model::MLP, x::AbstractMatrix, y::Flux.OneHotArray, mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(x))[mask] .== onecold(y)[mask])
end

# ╔═╡ b18384fe-b8ae-4f51-bd73-d129d5e70f98
md"""
After training the model, we can call the `accuracy` function to see how well our model performs on unseen labels.
Here, we are interested in the accuracy of the model, *i.e.*, the ratio of correctly classified nodes:
"""

# ╔═╡ 54a2972e-b107-47c8-bf7e-eb51b4ccbe02
md"""
As one can see, our MLP performs rather bad with only about 47% test accuracy.
But why does the MLP do not perform better?
The main reason for that is that this model suffers from heavy overfitting due to only having access to a **small amount of training nodes**, and therefore generalizes poorly to unseen node representations.

It also fails to incorporate an important bias into the model: **Cited papers are very likely related to the category of a document**.
That is exactly where Graph Neural Networks come into play and can help to boost the performance of our model.
"""

# ╔═╡ 623e7b53-046c-4858-89d9-13caae45255d
md"""
## Training a Graph Convolutional Neural Network (GNN)

We can easily convert our MLP to a GNN by swapping the `torch.nn.Linear` layers with PyG's GNN operators.

Following-up on [the first part of this tutorial](), we replace the linear layers by the [`GCNConv`]() module.
To recap, the **GCN layer** ([Kipf et al. (2017)](https://arxiv.org/abs/1609.02907)) is defined as

```math
\mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \sum_{w \in \mathcal{N}(v) \, \cup \, \{ v \}} \frac{1}{c_{w,v}} \cdot \mathbf{x}_w^{(\ell)}
```

where ``\mathbf{W}^{(\ell + 1)}`` denotes a trainable weight matrix of shape `[num_output_features, num_input_features]` and $c_{w,v}$ refers to a fixed normalization coefficient for each edge.
In contrast, a single `Linear` layer is defined as

```math
\mathbf{x}_v^{(\ell + 1)} = \mathbf{W}^{(\ell + 1)} \mathbf{x}_v^{(\ell)}
```

which does not make use of neighboring node information.
"""

# ╔═╡ eb36a46c-f139-425e-8a93-207bc4a16f89
begin 
    struct GCN
        layers::NamedTuple
    end
    
    Flux.@functor GCN # provides parameter collection, gpu movement and more



    function GCN(num_features, num_classes, hidden_channels; drop_rate=0.5)
        layers = (conv1 = GCNConv(num_features => hidden_channels),
                    drop = Dropout(drop_rate), 
                    conv2 = GCNConv(hidden_channels => num_classes))
        return GCN(layers)
    end

    function (gcn::GCN)(g::GNNGraph, x::AbstractMatrix)
        l = gcn.layers
        x = l.conv1(g, x)
        x = relu.(x)
        x = l.drop(x)
        x = l.conv2(g, x)
        return x
    end
end

# ╔═╡ 20b5f802-abce-49e1-a442-f381e80c0f85
md"""
Now let's visualize the node embeddings of our **untrained** GCN network.
"""

# ╔═╡ b295adce-b37e-45f3-963a-3699d714e36d
# ╠═╡ show_logs = false
begin
    gcn = GCN(num_features, num_classes, hidden_channels)
    h_untrained = gcn(g, x) |> transpose
    visualize_tsne(h_untrained, g.ndata.targets)
end

# ╔═╡ 5538970f-b273-4122-9d50-7deb049e6934
md"""
We certainly can do better by training our model.
The training and testing procedure is once again the same, but this time we make use of the node features `x` **and** the graph `g` as input to our GCN model.
"""

# ╔═╡ 901d9478-9a12-4122-905d-6cfc6d80e84c
function train(model::GCN, g::GNNGraph, x::AbstractMatrix, epochs::Int, ps, opt)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, gs = Flux.withgradient(ps) do
            ŷ = model(g, x)
            logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
        end
    
        Flux.Optimise.update!(opt, ps, gs)
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end


# ╔═╡ 026911dd-6a27-49ce-9d41-21e01646c10a
# ╠═╡ show_logs = false
begin
    mlp = MLP(num_features, num_classes, hidden_channels)
    ps_mlp = Flux.params(mlp)
    opt_mlp = ADAM(1e-3)
    epochs = 2000
    train(mlp, g.ndata.features, epochs, opt_mlp, ps_mlp)
end

# ╔═╡ 65d9fd3d-1649-4b95-a106-f26fa4ab9bce
function accuracy(model::GCN, g::GNNGraph, x::AbstractMatrix, y::Flux.OneHotArray, mask::BitVector)
    Flux.testmode!(model)
    mean(onecold(model(g, x))[mask] .== onecold(y)[mask])
end

# ╔═╡ b2302697-1e20-4721-ae93-0b121ff9ce8f
accuracy(mlp, g.ndata.features, y, .!train_mask)

# ╔═╡ 20be52b1-1c33-4f54-b5c0-fecc4e24fbb5
# ╠═╡ show_logs = false
begin
    ps_gcn = Flux.params(gcn)
    opt_gcn = ADAM(1e-2)
    train(gcn, g, x, epochs, ps_gcn, opt_gcn)
end

# ╔═╡ 5aa99aff-b5ed-40ec-a7ec-0ba53385e6bd
md"""
Now let's evaluate the loss of our trained GCN.
"""

# ╔═╡ 2163d0d8-0661-4d11-a09e-708769011d35
with_terminal() do
    train_accuracy = accuracy(gcn, g, g.ndata.features, y, train_mask)
    test_accuracy = accuracy(gcn, g, g.ndata.features, y,  .!train_mask)
    
    println("Train accuracy: $(train_accuracy)")
    println("Test accuracy: $(test_accuracy)")
end

# ╔═╡ 6cd49f3f-a415-4b6a-9323-4d6aa6b87f18
md"""
**There it is!**
By simply swapping the linear layers with GNN layers, we can reach **75.77% of test accuracy**!
This is in stark contrast to the 59% of test accuracy obtained by our MLP, indicating that relational information plays a crucial role in obtaining better performance.

We can also verify that once again by looking at the output embeddings of our trained model, which now produces a far better clustering of nodes of the same category.
"""

# ╔═╡ 7a93a802-6774-42f9-b6da-7ae614464e72
# ╠═╡ show_logs = false
begin
    Flux.testmode!(gcn) # inference mode

    out_trained = gcn(g, x) |> transpose
    visualize_tsne(out_trained, g.ndata.targets)
end

# ╔═╡ 50a409fd-d80b-4c48-a51b-173c39a6dcb4
md"""
## (Optional) Exercises

1. To achieve better model performance and to avoid overfitting, it is usually a good idea to select the best model based on an additional validation set.
The `Cora` dataset provides a validation node set as `g.ndata.val_mask`, but we haven't used it yet.
Can you modify the code to select and test the model with the highest validation performance?
This should bring test performance to **82% accuracy**.

2. How does `GCN` behave when increasing the hidden feature dimensionality or the number of layers?
Does increasing the number of layers help at all?

3. You can try to use different GNN layers to see how model performance changes. What happens if you swap out all `GCNConv` instances with [`GATConv`](https://carlolucibello.github.io/GraphNeuralNetworks.jl/dev/api/conv/#GraphNeuralNetworks.GATConv) layers that make use of attention? Try to write a 2-layer `GAT` model that makes use of 8 attention heads in the first layer and 1 attention head in the second layer, uses a `dropout` ratio of `0.6` inside and outside each `GATConv` call, and uses a `hidden_channels` dimensions of `8` per head.
"""

# ╔═╡ c343419f-a1d7-45a0-b600-2c868588b33a
md"""
## Conclusion
In this tutorial, we have seen how to apply GNNs to real-world problems, and, in particular, how they can effectively be used for boosting a model's performance. In the next section, we will look into how GNNs can be used for the task of graph classification.

[Next tutorial: Graph Classification with Graph Neural Networks]()
"""

# ╔═╡ Cell order:
# ╟─2c710e0f-4275-4440-a3a9-27eabf61823a
# ╟─ca2f0293-7eac-4d9a-9a2f-fda47fd95a99
# ╟─4455f18c-2bd9-42ed-bce3-cfe6561eab23
# ╠═5463330a-0161-11ed-1b18-936030a32bbf
# ╟─0d556a7c-d4b6-4cef-806c-3e1712de0791
# ╠═997b5387-3811-4998-a9d1-7981b58b9e09
# ╟─4b6fa18d-7ccd-4c07-8dc3-ded4d7da8562
# ╠═edab1e3a-31f6-471f-9835-5b1f97e5cf3f
# ╟─d73a2db5-9417-4b2c-a9f5-b7d499a53fcb
# ╠═32bb90c1-c802-4c0c-a620-5d3b8f3f2477
# ╟─3438ee7f-bfca-465d-85df-13379622d415
# ╠═eec6fb60-0774-4f2a-bcb7-dbc28ab747a6
# ╟─bd2fd04d-7fb0-4b31-959b-bddabe681754
# ╠═b29c3a02-c21b-4b10-aa04-b90bcc2931d8
# ╠═16d9fbad-d4dc-4b51-9576-1736d228e2b3
# ╟─923d061c-25c3-4826-8147-9afa3dbd5bac
# ╠═28e00b95-56db-4d36-a205-fd24d3c54e17
# ╟─fa743000-604f-4d28-99f1-46ab2f884b8e
# ╠═f972f61b-2001-409b-9190-ac2c0652829a
# ╟─4dade64a-e28e-42c7-8ad5-93fc04724d4d
# ╠═05979cfe-439c-4abc-90cd-6ca2a05f6e0f
# ╠═a3f420e1-7521-4df9-b6d5-fc0a1fd05095
# ╠═026911dd-6a27-49ce-9d41-21e01646c10a
# ╟─b18384fe-b8ae-4f51-bd73-d129d5e70f98
# ╠═b2302697-1e20-4721-ae93-0b121ff9ce8f
# ╟─54a2972e-b107-47c8-bf7e-eb51b4ccbe02
# ╟─623e7b53-046c-4858-89d9-13caae45255d
# ╠═eb36a46c-f139-425e-8a93-207bc4a16f89
# ╟─20b5f802-abce-49e1-a442-f381e80c0f85
# ╠═b295adce-b37e-45f3-963a-3699d714e36d
# ╟─5538970f-b273-4122-9d50-7deb049e6934
# ╠═901d9478-9a12-4122-905d-6cfc6d80e84c
# ╠═65d9fd3d-1649-4b95-a106-f26fa4ab9bce
# ╠═20be52b1-1c33-4f54-b5c0-fecc4e24fbb5
# ╟─5aa99aff-b5ed-40ec-a7ec-0ba53385e6bd
# ╠═2163d0d8-0661-4d11-a09e-708769011d35
# ╟─6cd49f3f-a415-4b6a-9323-4d6aa6b87f18
# ╠═7a93a802-6774-42f9-b6da-7ae614464e72
# ╟─50a409fd-d80b-4c48-a51b-173c39a6dcb4
# ╟─c343419f-a1d7-45a0-b600-2c868588b33a