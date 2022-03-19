using GraphNeuralNetworks, Test, Flux
using BenchmarkTools
using ProfileView

g = rand_graph(10, 30, ndata=rand(Float32, 2, 10))
l = GATConv(2 => 2)
y = l(g, g.ndata.x)
@assert eltype(y) == Float32

dx = gradient(x -> sum(sin.(l(g, x))), g.ndata.x)[1]
@assert eltype(dx) == Float32

struct B
    slope
end

(a::B)(x) = leakyrelu(x, a.slope)

a = B(0.3f0)
grad = gradient(a -> a(0.1f0), a)[1]
