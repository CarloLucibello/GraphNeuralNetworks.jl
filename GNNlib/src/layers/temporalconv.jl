####################### TGCN ######################################

function tgcn_conv(l, h, g::GNNGraph, x::AbstractArray)
    x̃ = l.conv(g, x)
    h, x̃ = l.gru(h, x̃)
    return h, x̃
end


function a3tgcn_conv(a3tgcn, g::GNNGraph, x::AbstractArray)
    h = a3tgcn.tgcn(g, x)
    e = a3tgcn.dense1(h)
    e = a3tgcn.dense2(e)
    a = softmax(e, dims = 3)
    c = sum(a .* h , dims = 3)
    if length(size(c)) == 3
        c = dropdims(c, dims = 3)
    end
    return c
end

