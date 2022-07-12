### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 2c710e0f-4275-4440-a3a9-27eabf61823a
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate(; temp=true)
    packages = [
        PackageSpec(; path=joinpath(@__DIR__,"..","..","..")),
        PackageSpec(; name="Flux", version="0.13"),
		PackageSpec(; name="MLDatasets", version="0.7"),
		PackageSpec(; name="Plots"),
		PackageSpec(; name="TSne"),
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
	using TSne
	using Random
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
	Random.seed!(17) # for reproducibility
end

# ╔═╡ 0d556a7c-d4b6-4cef-806c-3e1712de0791
md"""
## Visualize
"""

# ╔═╡ 997b5387-3811-4998-a9d1-7981b58b9e09
function visualize_tsne(out, targets)
    z = tsne(out, 2)
    scatter(z[:, 1], z[:, 2], color=Int.(targets[1:size(z,1)]))
end

# ╔═╡ 4b6fa18d-7ccd-4c07-8dc3-ded4d7da8562
md"""
## Dataset: Cora
"""

# ╔═╡ edab1e3a-31f6-471f-9835-5b1f97e5cf3f
dataset = Cora()

# ╔═╡ 32bb90c1-c802-4c0c-a620-5d3b8f3f2477
dataset.metadata

# ╔═╡ eec6fb60-0774-4f2a-bcb7-dbc28ab747a6
dataset[:]

# ╔═╡ b29c3a02-c21b-4b10-aa04-b90bcc2931d8
g = mldataset2gnngraph(dataset)

# ╔═╡ 28e00b95-56db-4d36-a205-fd24d3c54e17
begin
	x = g.ndata.features
	y = onehotbatch(g.ndata.targets, 1:7)
	train_mask = g.ndata.train_mask
end

# ╔═╡ 4cc720bf-200a-4028-ad60-d9bdce856f47
begin
	num_features = size(x)[1]
	hidden_channels = 16
	num_classes = dataset.metadata["num_classes"]
end

# ╔═╡ fa743000-604f-4d28-99f1-46ab2f884b8e
md"""
## MLP
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
### Train
"""

# ╔═╡ 05979cfe-439c-4abc-90cd-6ca2a05f6e0f
function train(model::MLP, data::AbstractMatrix, epochs::Int, opt, ps)
    Flux.trainmode!(model)

    for epoch in 1:epochs
        loss, gs = Flux.withgradient(ps) do
            ŷ = model(data)
            logitcrossentropy(ŷ[:,train_mask], y[:,train_mask])
        end
    
        Flux.Optimise.update!(opt, ps, gs)
        if epoch % 200 == 0
            @show epoch, loss
        end
    end
end

# ╔═╡ 623e7b53-046c-4858-89d9-13caae45255d
md"""
## GCN
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
	opt_mlp = ADAM(1e-2)
	epochs = 2000
	train(mlp, g.ndata.features, epochs, opt_mlp, ps_mlp)
end

# ╔═╡ b295adce-b37e-45f3-963a-3699d714e36d
# ╠═╡ show_logs = false
begin
	gcn = GCN(num_features, num_classes, hidden_channels)
	h_untrained = gcn(g, x) |> transpose
	visualize_tsne(h_untrained, g.ndata.targets)
end

# ╔═╡ 20be52b1-1c33-4f54-b5c0-fecc4e24fbb5
# ╠═╡ show_logs = false
begin
	ps_gcn = Flux.params(gcn)
	opt_gcn = ADAM(1e-2)
	train(gcn, g, x, epochs, ps_gcn, opt_gcn)
end

# ╔═╡ 7a93a802-6774-42f9-b6da-7ae614464e72
# ╠═╡ show_logs = false
begin
	Flux.testmode!(gcn)
	h_trained = gcn(g, x) |> transpose
	visualize_tsne(h_trained, g.ndata.targets)
end

# ╔═╡ Cell order:
# ╠═2c710e0f-4275-4440-a3a9-27eabf61823a
# ╠═5463330a-0161-11ed-1b18-936030a32bbf
# ╟─0d556a7c-d4b6-4cef-806c-3e1712de0791
# ╠═997b5387-3811-4998-a9d1-7981b58b9e09
# ╟─4b6fa18d-7ccd-4c07-8dc3-ded4d7da8562
# ╠═edab1e3a-31f6-471f-9835-5b1f97e5cf3f
# ╠═32bb90c1-c802-4c0c-a620-5d3b8f3f2477
# ╠═eec6fb60-0774-4f2a-bcb7-dbc28ab747a6
# ╠═b29c3a02-c21b-4b10-aa04-b90bcc2931d8
# ╠═28e00b95-56db-4d36-a205-fd24d3c54e17
# ╠═4cc720bf-200a-4028-ad60-d9bdce856f47
# ╟─fa743000-604f-4d28-99f1-46ab2f884b8e
# ╠═f972f61b-2001-409b-9190-ac2c0652829a
# ╟─4dade64a-e28e-42c7-8ad5-93fc04724d4d
# ╠═05979cfe-439c-4abc-90cd-6ca2a05f6e0f
# ╠═026911dd-6a27-49ce-9d41-21e01646c10a
# ╠═623e7b53-046c-4858-89d9-13caae45255d
# ╠═eb36a46c-f139-425e-8a93-207bc4a16f89
# ╠═901d9478-9a12-4122-905d-6cfc6d80e84c
# ╠═b295adce-b37e-45f3-963a-3699d714e36d
# ╠═20be52b1-1c33-4f54-b5c0-fecc4e24fbb5
# ╠═7a93a802-6774-42f9-b6da-7ae614464e72
