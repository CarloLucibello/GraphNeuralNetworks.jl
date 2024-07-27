@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    g = rand_graph(10, 40, seed=1234)
    x = randn(rng, Float32, 3, 10)

    @testset "GCNConv" begin
        l = GCNConv(3 => 5, relu)
        @test l isa GNNLayer
        ps = Lux.initialparameters(rng, l)
        st = Lux.initialstates(rng, l)
        @test Lux.parameterlength(l) == Lux.parameterlength(ps)
        @test Lux.statelength(l) == Lux.statelength(st)

        y, _ = l(g, x, ps, st)
        @test Lux.outputsize(l) == (5,)
        @test size(y) == (5, 10)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        @eval @test_gradients $loss $x $ps atol=1.0f-3 rtol=1.0f-3 skip_tracker=true
    end

    @testset "ChebConv" begin
        l = ChebConv(3 => 5, 2)
        @test l isa GNNLayer
        ps = Lux.initialparameters(rng, l)
        st = Lux.initialstates(rng, l)
        @test Lux.parameterlength(l) == Lux.parameterlength(ps)
        @test Lux.statelength(l) == Lux.statelength(st)

        y, _ = l(g, x, ps, st)
        @test Lux.outputsize(l) == (5,)
        @test size(y) == (5, 10)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        @eval @test_gradients $loss $x $ps atol=1.0f-3 rtol=1.0f-3 skip_tracker=true skip_reverse_diff=true
    end

    @testset "GraphConv" begin
        l = GraphConv(3 => 5, relu)
        @test l isa GNNLayer
        ps = Lux.initialparameters(rng, l)
        st = Lux.initialstates(rng, l)
        @test Lux.parameterlength(l) == Lux.parameterlength(ps)
        @test Lux.statelength(l) == Lux.statelength(st)

        y, _ = l(g, x, ps, st)
        @test Lux.outputsize(l) == (5,)
        @test size(y) == (5, 10)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        @eval @test_gradients $loss $x $ps atol=1.0f-3 rtol=1.0f-3 skip_tracker=true
    end

    @testset "AGNNConv" begin
        l = AGNNConv(init_beta=1.0f0)
        @test l isa GNNLayer
        ps = Lux.initialparameters(rng, l)
        st = Lux.initialstates(rng, l)
        @test Lux.parameterlength(ps) == 1
        @test Lux.parameterlength(l) == Lux.parameterlength(ps)
        @test Lux.statelength(l) == Lux.statelength(st)

        y, _ = l(g, x, ps, st)
        @test size(y) == size(x)
        loss = (x, ps) -> sum(first(l(g, x, ps, st)))
        @eval @test_gradients $loss $x $ps atol=1.0f-3 rtol=1.0f-3 skip_tracker=true skip_reverse_diff=true
    end
end
