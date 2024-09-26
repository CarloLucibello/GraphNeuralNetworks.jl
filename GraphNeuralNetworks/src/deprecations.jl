
# V1.0 deprecations 
# TODO doe some reason this is not working
# @deprecate (l::GCNConv)(g, x, edge_weight, norm_fn; conv_weight=nothing)  l(g, x, edge_weight; norm_fn, conv_weight)
# @deprecate (l::GNNLayer)(gs::AbstractVector{<:GNNGraph}, args...; kws...) l(MLUtils.batch(gs), args...; kws...)