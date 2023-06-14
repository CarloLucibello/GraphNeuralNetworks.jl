# Load the packages
using GraphNeuralNetworks, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux
using Statistics: mean
using MLDatasets
using CUDA
# CUDA.allowscalar(false) # Some scalar indexing is still done by DiffEqFlux

# device = cpu # `gpu` not working yet
device = CUDA.functional() ? gpu : cpu

# LOAD DATA
X, y = MNIST(:train)[:]
y = onehotbatch(y, 0:9)

# Define the Neural GDE
diffeqsol_to_array(x) = reshape(device(x), size(x)[1:2])

nin, nhidden, nout = 28 * 28, 100, 10
epochs = 10

node_chain = Chain(Dense(nhidden => nhidden, tanh),
                   Dense(nhidden => nhidden)) |> device

node = NeuralODE(node_chain,
                 (0.0f0, 1.0f0), Tsit5(), save_everystep = false,
                 reltol = 1e-3, abstol = 1e-3, save_start = false) |> device

model = Chain(Flux.flatten,
              Dense(nin => nhidden, relu),
              node,
              diffeqsol_to_array,
              Dense(nhidden, nout)) |> device

# # Training

# ## Optimizer
opt = Flux.setup(Adam(0.01), model)

function eval_loss_accuracy(X, y)
    ŷ = model(X)
    l = logitcrossentropy(ŷ, y)
    acc = mean(onecold(ŷ) .== onecold(y))
    return (loss = round(l, digits = 4), acc = round(acc * 100, digits = 2))
end

# ## Training Loop
for epoch in 1:epochs
    grad = gradient(model) do model
        ŷ = model(X)
        logitcrossentropy(ŷ, y)
    end
    Flux.update!(opt, model, grad[1])
    @show eval_loss_accuracy(X, y)
end
