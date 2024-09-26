# Load the packages
using GraphNeuralNetworks, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux
using Statistics: mean
using MLDatasets: Cora
using CUDA
# CUDA.allowscalar(false) # Some scalar indexing is still done by DiffEqFlux

# device = cpu # `gpu` not working yet
device = CUDA.functional() ? gpu : cpu

# LOAD DATA
dataset = Cora()
classes = dataset.metadata["classes"]
g = mldataset2gnngraph(dataset) |> device
X = g.ndata.features
y = onehotbatch(g.ndata.targets |> cpu, classes) |> device # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:, train_mask]

# Model and Data Configuration
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 40

# Define the Neural GDE
diffeqsol_to_array(x) = reshape(device(x), size(x)[1:2])

node_chain = GNNChain(GCNConv(nhidden => nhidden, relu),
                      GCNConv(nhidden => nhidden, relu)) |> device

node = NeuralODE(WithGraph(node_chain, g),
                 (0.0f0, 1.0f0), Tsit5(), save_everystep = false,
                 reltol = 1e-3, abstol = 1e-3, save_start = false) |> device

model = GNNChain(GCNConv(nin => nhidden, relu),
                 node,
                 diffeqsol_to_array,
                 Dense(nhidden, nout)) |> device

# # Training

opt = Flux.setup(Adam(0.01), model)

function eval_loss_accuracy(X, y, mask)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
end

# ## Training Loop
for epoch in 1:epochs
    grad = gradient(model) do model
        ŷ = model(g, X)
        logitcrossentropy(ŷ[:, train_mask], ytrain)
    end
    Flux.update!(opt, model, grad[1])
    @show eval_loss_accuracy(X, y, train_mask)
end
