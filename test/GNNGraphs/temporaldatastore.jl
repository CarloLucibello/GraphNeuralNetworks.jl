@testset "constructor" begin
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,5,3), :y => rand(3,3,5,2)))
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,4,3), :y => rand(3,3,5,3)))

    x = rand(10,5)
    @test  TemporalDataStore(5,1, (:x => x)) == DataStore(5, (:x => x))

    @testset "keyword args" begin
        tds = TemporalDataStore(4, 10, x = rand(2,4,10), y = rand(4, 10))
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)

        tds = TemporalDataStore(x = rand(2,4,10), y = rand(4, 10)) #possible feat: should it understand by itself n and t?
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)
    end
end

@testset "getdata / getn / gett" begin
    tds = TemporalDataStore(4, 10, x = rand(2,4,10))
    @test getdata(tds) == getfield(tds, :_data)
    @test_throws KeyError tds.data
    @test getn(tds) == getfield(tds, :_n)
    @test_throws KeyError tds.n
    @test gett(tds) == getfield(tds, :_t)
    @test_throws KeyError tds.t
end

@testset "getproperty / setproperty!" begin
    x = rand(10,5)
    z = rand(3,10,5)
    tds = TemporalDataStore(10, 5, (:x => x))
    @test tds.x == tds[:x] == x
    @test_throws DimensionMismatch tds.z=rand(10,4)
    tds.z = z
    @test tds.z == z
end


