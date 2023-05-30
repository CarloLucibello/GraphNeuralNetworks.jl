@testset "constructor" begin
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,5,3), :y => rand(3,3,5,2)))
    @test_throws AssertionError TemporalDataStore(5,3, (:x => rand(10,4,3), :y => rand(3,3,5,3)))

    xvalues = rand(10,5)
    @test  TemporalDataStore(5,1, x = xvalues) == DataStore(5, (:x => xvalues))

    @testset "keyword args" begin
        tds = TemporalDataStore(4, 10, x = rand(2,4,10), y = rand(4, 10))
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)

        tds = TemporalDataStore(x = rand(2,4,10), y = rand(4, 10)) #possible feat: should it understand by itself n and t?
        @test size(tds.x) == (2, 4, 10)
        @test size(tds.y) == (4, 10)
    end
end