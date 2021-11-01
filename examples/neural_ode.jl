# Load the packages
using GraphNeuralNetworks, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics: mean
using MLDatasets: Cora
using CUDA
# CUDA.allowscalar(false) # Some scalar indexing is still done by DiffEqFlux

# device = cpu # `gpu` not working yet
device = CUDA.functional() ? gpu : cpu

# LOAD DATA
data = Cora.dataset()
g = GNNGraph(data.adjacency_list) |> device
X = data.node_features |> device
y = onehotbatch(data.node_labels, 1:data.num_classes) |> device
train_ids = data.train_indices |> device
val_ids = data.val_indices |> device
test_ids = data.test_indices |> device
ytrain = y[:, train_ids]


# Model and Data Configuration
nin = size(X, 1)
nhidden = 16
nout = data.num_classes 
epochs = 40

# Define the Neural GDE
diffeqsol_to_array(x) = reshape(device(x), size(x)[1:2])

# GCNConv(nhidden => nhidden, graph=g),

node_chain = GNNChain(GCNConv(nhidden => nhidden, relu),
                      GCNConv(nhidden => nhidden, relu)) |> device

node = NeuralODE(WithGraph(node_chain, g),
                (0.f0, 1.f0), Tsit5(), save_everystep = false,
                reltol = 1e-3, abstol = 1e-3, save_start = false) |> device

model = GNNChain(GCNConv(nin => nhidden, relu),
                 Dropout(0.5),
                 node,
                 diffeqsol_to_array,
                 Dense(nhidden, nout)) |> device

# Loss
loss(x, y) = logitcrossentropy(model(g, x), y)
accuracy(x, y) = mean(onecold(model(g, x)) .== onecold(y))

# # Training
# ## Model Parameters
ps = Flux.params(model);

# ## Optimizer
opt = ADAM(0.01)

# ## Training Loop
for epoch in 1:epochs
    gs = gradient(() -> loss(X, y), ps)
    Flux.Optimise.update!(opt, ps, gs)
    @show(accuracy(X, y))
end
