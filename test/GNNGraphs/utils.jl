@testset "Utils" begin
    @testset "edge encoding/decoding" begin
        # not is_bidirected
        n = 5
        s = [1,1,2,3,3,4,5]
        t = [1,3,1,1,2,5,5]
        
        # directed=true
        idx, maxid = GNNGraphs.edge_encoding(s, t, n) 
        @test maxid == n^2
        @test idx == [1, 3, 6, 11, 12, 20, 25]

        sdec, tdec = GNNGraphs.edge_decoding(idx, n) 
        @test sdec == s
        @test tdec == t


        # directed=false
        idx, maxid = GNNGraphs.edge_encoding(s, t, n, directed=false)
        @test maxid == n * (n+1)÷2
        @test idx == [1, 3, 2, 3, 7, 14, 15]
    end
end