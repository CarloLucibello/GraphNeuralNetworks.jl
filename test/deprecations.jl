@testset "deprecations" begin
    @testset "propagate" begin
        struct GCN{A<:AbstractMatrix, B, F} <: GNNLayer
            weight::A
            bias::B
            σ::F
        end

        Flux.@functor GCN # allow collecting params, gpu movement, etc...

        function GCN(ch::Pair{Int,Int}, σ=identity)
            in, out = ch
            W = Flux.glorot_uniform(out, in)
            b = zeros(Float32, out)
            GCN(W, b, σ)
        end

        GraphNeuralNetworks.compute_message(l::GCN, xi, xj, e) = xj 

        function (l::GCN)(g::GNNGraph, x::AbstractMatrix{T}) where T
            x, _ = propagate(l, g, +, x) 
            return l.σ.(l.weight * x .+ l.bias)
        end

        function new_forward(l, g, x)
            x = propagate(copy_xj, g, +, xj=x) 
            return l.σ.(l.weight * x .+ l.bias)
        end

        g = GNNGraph(random_regular_graph(10, 4), ndata=randn(3, 10))
        l = GCN(3 => 5, tanh)
        @test l(g, g.ndata.x) ≈ new_forward(l, g, g.ndata.x)
    end
end
