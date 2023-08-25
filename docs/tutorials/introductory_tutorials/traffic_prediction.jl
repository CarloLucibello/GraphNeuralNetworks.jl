### A Pluto.jl notebook ###
# v0.19.26

#> [frontmatter]
#> author = "[Aurora Rossi](https://github.com/aurorarossi)"
#> title = "Traffic Prediction using recurrent Temporal Graph Convolutional Network"
#> date = "2023-08-21"
#> description = "Traffic Prediction using GraphNeuralNetworks.jl"
#> cover = "assets/traffic.gif"

using Markdown
using InteractiveUtils

# ╔═╡ 177a835d-e91d-4f5f-8484-afb2abad400c
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.develop("GraphNeuralNetworks")
	Pkg.add("MLDatasets")
	Pkg.add("Plots")
end

# ╔═╡ 1f95ad97-a007-4724-84db-392b0026e1a4
begin
using GraphNeuralNetworks
using Flux
using Flux.Losses: mae
using MLDatasets: METRLA
using Statistics
using Plots
end

# ╔═╡ 5fdab668-4003-11ee-33f5-3953225b0c0f
md"
In this tutorial, we will learn how to use a recurrent Temporal Graph Convolutional Network (TGCN) to predict traffic in a spatio-temporal setting. Traffic forecasting is the problem of predicting future traffic trends on a road network given historical traffic data, such as, in our case, traffic speed and time of day.
"

# ╔═╡ 3dd0ce32-2339-4d5a-9a6f-1f662bc5500b
md"
## Import

We start by importing the necessary libraries. We use `GraphNeuralNetworks.jl`, `Flux.jl` and `MLDatasets.jl`, among others.
"

# ╔═╡ ec5caeb6-1f95-4cb9-8739-8cadba29a22d
md"
## Dataset: METR-LA

We use the `METR-LA` dataset from the paper [Diffusion Convolutional Recurrent Neural Network: Data-driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926.pdf), which contains traffic data from loop detectors in the highway of Los Angeles County. The dataset contains traffic speed data from March 1, 2012 to June 30, 2012. The data is collected every 5 minutes, resulting in 12 observations per hour, from 207 sensors. Each sensor is a node in the graph, and the edges represent the distances between the sensors.
"

# ╔═╡ f531e39c-6842-494a-b4ac-8904321098c9
dataset_metrla = METRLA(; num_timesteps = 3)

# ╔═╡ d5ebf9aa-cec8-4417-baaf-f2e8e19f1cad
 g = dataset_metrla[1]

# ╔═╡ dc2d5e98-2201-4754-bfc6-8ed2bbb82153
md"
`edge_data` contains the weights of the edges of the graph and
`node_data` contains a node feature vector and a target vector. The latter vectors contain batches of dimension `num_timesteps`, which means that they contain vectors with the node features and targets of `num_timesteps` time steps. Two consecutive batches are shifted by one-time step.
The node features are the traffic speed of the sensors and the time of the day, and the targets are the traffic speed of the sensors in the next time step.
Let's see some examples:
"

# ╔═╡ 0dde5fd3-72d0-4b15-afb3-9a5b102327c9
size(g.node_data.features[1])

# ╔═╡ f7a6d572-28cf-4d69-a9be-d49f367eca37
md"
The first dimension correspond to the two features (first line the speed value and the second line the time of the day), the second to the nodes and the third to the number of timestep `num_timesteps`.
"

# ╔═╡ 3d5503bc-bb97-422e-9465-becc7d3dbe07
size(g.node_data.targets[1])

# ╔═╡ 3569715d-08f5-4605-b946-9ef7ccd86ae5
md"
In the case of the targets the first dimension is 1 because they store just the speed value.
"

# ╔═╡ aa4eb172-2a42-4c01-a6ef-c6c95208d5b2
g.node_data.features[1][:,1,:]

# ╔═╡ 367ed417-4f53-44d4-8135-0c91c842a75f
g.node_data.features[2][:,1,:]

# ╔═╡ 7c084eaa-655c-4251-a342-6b6f4df76ddb
g.node_data.targets[1][:,1,:]

# ╔═╡ bf0d820d-32c0-4731-8053-53d5d499e009
function plot_data(data,sensor)
	p = plot(legend=false, xlabel="Time (h)", ylabel="Normalized speed")
	plotdata = []
	for i in 1:3:length(data)
		push!(plotdata,data[i][1,sensor,:])
	end
	plotdata = reduce(vcat,plotdata)
	plot!(p, collect(1:length(data)), plotdata, color = :green, xticks =([i for i in 0:50:250], ["$(i)" for i in 0:4:24]))
	return p
end

# ╔═╡ cb89d1a3-b4ff-421a-8717-a0b7f21dea1a
plot_data(g.node_data.features[1:288],1)

# ╔═╡ 3b49a612-3a04-4eb5-bfbc-360614f4581a
md"
Now let's construct the static graph, the temporal features and targets from the dataset.
"

# ╔═╡ 95d8bd24-a40d-409f-a1e7-4174428ef860
begin
graph = GNNGraph(g.edge_index; edata = g.edge_data, g.num_nodes)
features = g.node_data.features
targets = g.node_data.targets
end;  

# ╔═╡ fde2ac9e-b121-4105-8428-1820b9c17a43
md"
Now let's construct the `train_loader` and `data_loader`.
"


# ╔═╡ 111b7d5d-c7e3-44c0-9e5e-2ed1a86854d3
begin
train_loader = zip(features[1:200], targets[1:200])
test_loader = zip(features[2001:2288], targets[2001:2288])
end;

# ╔═╡ 572a6633-875b-4d7e-9afc-543b442948fb
md"
## Model: T-GCN

We use the T-GCN model from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction] (https://arxiv.org/pdf/1811.05320.pdf), which consists of a graph convolutional network (GCN) and a gated recurrent unit (GRU). The GCN is used to capture spatial features from the graph, and the GRU is used to capture temporal features from the feature time series.
"

# ╔═╡ 5502f4fa-3201-4980-b766-2ab88b175b11
model = GNNChain(TGCN(2 => 100), Dense(100, 1))

# ╔═╡ 4a1ec34a-1092-4b4a-b8a8-bd91939ffd9e
md"
![](https://www.researchgate.net/profile/Haifeng-Li-3/publication/335353434/figure/fig4/AS:851870352437249@1580113127759/The-architecture-of-the-Gated-Recurrent-Unit-model.jpg)
"

# ╔═╡ 755a88c2-c2e5-46d1-9582-af4b2c5a6bbd
md"
## Training

We train the model for 100 epochs, using the Adam optimizer with a learning rate of 0.001. We use the mean absolute error (MAE) as the loss function.
"

# ╔═╡ e83253b2-9f3a-44e2-a747-cce1661657c4
function train(graph, train_loader, model)

    opt = Flux.setup(Adam(0.001), model)

    for epoch in 1:100
        for (x, y) in train_loader
            x, y = (x, y)
            grads = Flux.gradient(model) do model
                ŷ = model(graph, x)
                Flux.mae(ŷ, y) 
            end
            Flux.update!(opt, model, grads[1])
		end
		
		if epoch % 10 == 0
			loss = mean([Flux.mae(model(graph,x), y) for (x, y) in train_loader])
			@show epoch, loss
		end
		
    end
    return model
end

# ╔═╡ 85a923da-3027-4f71-8db6-96852c115c03
train(graph, train_loader, model)

# ╔═╡ 39c82234-97ea-48d6-98dd-915f072b7f85
function plot_predicted_data(graph,features,targets, sensor)
	p = plot(legend=false, xlabel="Time (h)", ylabel="Normalized speed")
	prediction = []
	grand_truth =[]
	for i in 1:3:length(features)
		push!(grand_truth,targets[i][1,sensor,:])
		push!(prediction, model(graph, features[i])[1,sensor,:]) 
	end
	prediction = reduce(vcat,prediction)
	grand_truth = reduce(vcat, grand_truth)
	plot!(p, collect(1:length(features)), grand_truth, color = :blue, label = "Grand Truth", xticks =([i for i in 0:50:250], ["$(i)" for i in 0:4:24]))
	plot!(p, collect(1:length(features)), prediction, color = :red, label= "Prediction")
	return p
end

# ╔═╡ 8c3a903b-2c8a-4d4f-8eef-74d5611f2ce4
plot_predicted_data(graph,features[301:588],targets[301:588], 1)

# ╔═╡ 2c5f6250-ee7a-41b1-9551-bcfeba83ca8b
accuracy(ŷ, y) = 1 - Statistics.norm(y-ŷ)/Statistics.norm(y)

# ╔═╡ 1008dad4-d784-4c38-a7cf-d9b64728e28d
mean([accuracy(model(graph,x), y) for (x, y) in test_loader])

# ╔═╡ 8d0e8b9f-226f-4bff-9deb-046e6a897b71
md"The accuracy is not very good but can be improved by training using more data. We used a small subset of the dataset for this tutorial because of the computational cost of training the model. From the plot of the predictions, we can see that the model is able to capture the general trend of the traffic speed, but it is not able to capture the peaks of the traffic."

# ╔═╡ a7e4bb23-6687-476a-a0c2-1b2736873d9d
md"
## Conclusion

In this tutorial, we learned how to use a recurrent temporal graph convolutional network to predict traffic in a spatio-temporal setting. We used the TGCN model, which consists of a graph convolutional network (GCN) and a gated recurrent unit (GRU). We then trained the model for 100 epochs on a small subset of the METR-LA dataset. The accuracy of the model is not very good, but it can be improved by training on more data.
"

# ╔═╡ Cell order:
# ╟─5fdab668-4003-11ee-33f5-3953225b0c0f
# ╠═177a835d-e91d-4f5f-8484-afb2abad400c
# ╟─3dd0ce32-2339-4d5a-9a6f-1f662bc5500b
# ╠═1f95ad97-a007-4724-84db-392b0026e1a4
# ╟─ec5caeb6-1f95-4cb9-8739-8cadba29a22d
# ╠═f531e39c-6842-494a-b4ac-8904321098c9
# ╠═d5ebf9aa-cec8-4417-baaf-f2e8e19f1cad
# ╟─dc2d5e98-2201-4754-bfc6-8ed2bbb82153
# ╠═0dde5fd3-72d0-4b15-afb3-9a5b102327c9
# ╟─f7a6d572-28cf-4d69-a9be-d49f367eca37
# ╠═3d5503bc-bb97-422e-9465-becc7d3dbe07
# ╟─3569715d-08f5-4605-b946-9ef7ccd86ae5
# ╠═aa4eb172-2a42-4c01-a6ef-c6c95208d5b2
# ╠═367ed417-4f53-44d4-8135-0c91c842a75f
# ╠═7c084eaa-655c-4251-a342-6b6f4df76ddb
# ╠═bf0d820d-32c0-4731-8053-53d5d499e009
# ╠═cb89d1a3-b4ff-421a-8717-a0b7f21dea1a
# ╟─3b49a612-3a04-4eb5-bfbc-360614f4581a
# ╠═95d8bd24-a40d-409f-a1e7-4174428ef860
# ╟─fde2ac9e-b121-4105-8428-1820b9c17a43
# ╠═111b7d5d-c7e3-44c0-9e5e-2ed1a86854d3
# ╟─572a6633-875b-4d7e-9afc-543b442948fb
# ╠═5502f4fa-3201-4980-b766-2ab88b175b11
# ╟─4a1ec34a-1092-4b4a-b8a8-bd91939ffd9e
# ╟─755a88c2-c2e5-46d1-9582-af4b2c5a6bbd
# ╠═e83253b2-9f3a-44e2-a747-cce1661657c4
# ╠═85a923da-3027-4f71-8db6-96852c115c03
# ╠═39c82234-97ea-48d6-98dd-915f072b7f85
# ╠═8c3a903b-2c8a-4d4f-8eef-74d5611f2ce4
# ╠═2c5f6250-ee7a-41b1-9551-bcfeba83ca8b
# ╠═1008dad4-d784-4c38-a7cf-d9b64728e28d
# ╟─8d0e8b9f-226f-4bff-9deb-046e6a897b71
# ╟─a7e4bb23-6687-476a-a0c2-1b2736873d9d
