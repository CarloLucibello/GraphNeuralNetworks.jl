using Flux
using Flux: @epochs, mse, params

function Graphormer(input_dim, hidden_dim, output_dim, num_layers, num_heads)
    
    # Define the transformer encoder block
    function TransformerEncoder(hidden_dim, num_heads)
        multi_head_attention = Chain([Dense(hidden_dim, hidden_dim) for i in 1:num_heads]...)
        layer_norm1 = LayerNorm(hidden_dim)
        position_wise_feed_forward = Chain(Dense(hidden_dim, hidden_dim, relu), Dense(hidden_dim, hidden_dim))
        layer_norm2 = LayerNorm(hidden_dim)
        
        function (x)
            # Calculate multi-head attention
            heads = [head(x) for head in multi_head_attention]
            concatenated = Flux.cat(heads..., dims=3)
            attention_out = Flux.squeeze(sum(concatenated .* x, dims=2), dims=2)
            attention_out = layer_norm1(x + attention_out)
            
            # Calculate position-wise feed forward network
            ff_out = position_wise_feed_forward(attention_out)
            ff_out = layer_norm2(attention_out + ff_out)
            return ff_out
        end
    end
    
    input_embedding = Dense(input_dim, hidden_dim)
    transformer_layers = Chain([TransformerEncoder(hidden_dim, num_heads) for i in 1:num_layers]...)
    output_layer = Dense(hidden_dim, output_dim)
    
    function (x)
        x = input_embedding(x)
        x = transformer_layers(x)
        x = mean(x, dims=1)
        x = output_layer(x)
        return x
    end
end
