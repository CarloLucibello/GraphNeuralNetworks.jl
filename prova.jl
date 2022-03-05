using Flux
using GraphNeuralNetworks

function test(g)
    loader = Flux.DataLoader(g, batchsize=100, shuffle=true)
    l = collect(loader)
    return l
end

n = 5000
s = 10
x1 = Flux.batch([rand_graph(s, s, ndata = rand(1, s)) for i in 1:n]) 
x2 = Flux.batch([rand(s + s + s + s) for i in 1:n]) #source+target+data+extra
@btime test(x1)
@btime test(x2)
