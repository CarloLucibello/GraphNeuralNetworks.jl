
@testset "constructor" begin
    @test_throws AssertionError DataStore(10, (:x => rand(10), :y => rand(2, 4)))

    @testset "keyword args" begin
        ds = DataStore(10, x = rand(10), y = rand(2, 10))
        @test size(ds.x) == (10, )
        @test size(ds.y) == (2, 10)

        ds = DataStore(x = rand(10), y = rand(2, 10))
        @test size(ds.x) == (10, )
        @test size(ds.y) == (2, 10)
    end
end

@testset "getproperty / setproperty!" begin
    x = rand(10)
    ds = DataStore(10, (:x => x, :y => rand(2, 10)))
    @test ds.x == ds[:x] == x
    @test_throws AssertionError ds.z = rand(12)
    ds.z = [1:10;]
    @test ds.z == [1:10;]
end

@testset "map" begin
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))
    ds2 = map(x -> x .+ 1, ds)
    @test ds2.x == ds.x .+ 1
    @test ds2.y == ds.y .+ 1

    @test_throws AssertionError ds2 = map(x -> [x; x], ds)
end

@testset """getdata / getn""" begin
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))
    @test getdata(ds) == getfield(ds, :_data)
    @test_throws KeyError ds.data
    @test getn(ds) == getfield(ds, :_n)
    @test_throws KeyError ds.n
end

@testset "gradient" begin
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))
    
    f1(ds) = sum(ds.x)
    grad = gradient(f1, ds)[1]
    @test grad._data[:x] ≈ ngradient(f1, ds)[1][:x]

    g = rand_graph(5, 2)
    x = rand(2, 5)
    grad = gradient(x -> sum(exp, GNNGraph(g, ndata=x).ndata.x), x)[1]
    @test grad == exp.(x)
end

@testset "functor" begin 
    ds = DataStore(10, (:x => zeros(10), :y => ones(2, 10)))
    p, re = Functors.functor(ds)
    @test p[1] === getn(ds)
    @test p[2] === getdata(ds)
    @test ds == re(p)

    ds2 = Functors.fmap(ds) do x 
        if x isa AbstractArray
            x .+ 1
        else
            x
        end
    end
    @test ds isa DataStore
    @test ds2.x == ds.x .+ 1
end
