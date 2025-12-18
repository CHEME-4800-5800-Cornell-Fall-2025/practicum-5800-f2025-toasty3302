"""
    build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple) -> MyClassicalHopfieldNetworkModel

Build a Classical Hopfield Network model using Hebbian learning from memory patterns.

### Arguments
- `modeltype::Type{MyClassicalHopfieldNetworkModel}`: The type of model to construct
- `data::NamedTuple`: Named tuple containing:
    - `memories::Array{Int32,2}`: Matrix where each column is a memory pattern (N × K, where N = neurons, K = memories)

### Returns
- `MyClassicalHopfieldNetworkModel`: A Hopfield network with weights computed via Hebbian learning rule
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel
    
    # initialize -
    model = MyClassicalHopfieldNetworkModel()
    memories = data.memories  # N × K matrix (patterns in columns)
    
    # get dimensions -
    N = size(memories, 1)  # number of neurons (pixels)
    K = size(memories, 2)  # number of memories
    
    # initialize weight matrix and bias -
    W = zeros(Float32, N, N)
    b = zeros(Float32, N)
    
    # compute weights using Hebbian learning rule: W = (1/K) * Σ sᵢ ⊗ sᵢᵀ
    for k in 1:K
        sₖ = memories[:, k]  # get k-th memory pattern
        W .+= sₖ * sₖ'       # outer product and accumulate
    end
    W ./= K  # normalize by number of memories
    
    # set diagonal to zero (no self-connections)
    for i in 1:N
        W[i, i] = 0.0f0
    end
    
    # compute energy for each stored memory
    energy = Dict{Int64, Float32}()
    for k in 1:K
        sₖ = memories[:, k]
        # E(s) = -0.5 * s' * W * s - b' * s
        E = -0.5f0 * (sₖ' * W * sₖ) - (b' * sₖ)
        energy[k] = Float32(E)
    end
    
    # set model fields -
    model.W = W
    model.b = b
    model.energy = energy
    
    # return -
    return model
end