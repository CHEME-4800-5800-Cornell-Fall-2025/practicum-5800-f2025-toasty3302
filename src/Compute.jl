"""
    decode(s::Array{Int32,1}) -> Array{Float32,2}

Decode a binary state vector back to a 2D image matrix.

### Arguments
- `s::Array{Int32,1}`: A flattened binary state vector of length N² (values ∈ {-1, 1})

### Returns
- `Array{Float32,2}`: A √N × √N matrix where -1 → 0.0 (black) and 1 → 1.0 (white)
"""
function decode(s::Array{Int32,1})::Array{Float32,2}
    
    # get dimensions (assuming square image)
    N = length(s)
    n = round(Int, sqrt(N))  # side length of square image
    
    # initialize output matrix
    image = Array{Float32,2}(undef, n, n)
    
    # convert from vector to matrix form
    linearindex = 1
    for row in 1:n
        for col in 1:n
            # convert: -1 → 0.0 (black), 1 → 1.0 (white)
            if s[linearindex] == -1
                image[row, col] = 0.0f0
            else
                image[row, col] = 1.0f0
            end
            linearindex += 1
        end
    end
    
    return image
end

"""
    hamming(s1::Array{Int32,1}, s2::Array{Int32,1}) -> Int

Compute the Hamming distance between two binary state vectors.

### Arguments
- `s1::Array{Int32,1}`: First binary state vector
- `s2::Array{Int32,1}`: Second binary state vector

### Returns
- `Int`: Number of positions where the vectors differ
"""
function hamming(s1::Array{Int32,1}, s2::Array{Int32,1})::Int
    return sum(s1 .!= s2)
end

"""
    energy(model::MyClassicalHopfieldNetworkModel, s::Array{Int32,1}) -> Float32

Compute the energy of a given state in the Hopfield network.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: The Hopfield network model
- `s::Array{Int32,1}`: The state vector to evaluate

### Returns
- `Float32`: The energy value E(s) = -0.5 * s' * W * s - b' * s
"""
function energy(model::MyClassicalHopfieldNetworkModel, s::Array{Int32,1})::Float32
    W = model.W
    b = model.b
    sf = Float32.(s)  # convert to Float32 for computation
    E = -0.5f0 * (sf' * W * sf) - (b' * sf)
    return Float32(E)
end

"""
    recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, true_image_energy::Float32;
            maxiterations::Int64=1000, patience::Union{Int,Nothing}=nothing, 
            miniterations_before_convergence::Union{Int,Nothing}=nothing) -> Tuple{Dict, Dict}

Recover a memory from the Hopfield network using asynchronous update.

### Arguments
- `model::MyClassicalHopfieldNetworkModel`: The trained Hopfield network
- `sₒ::Array{Int32,1}`: The initial (corrupted) state vector
- `true_image_energy::Float32`: Energy of the true target memory (for convergence check)
- `maxiterations::Int64=1000`: Maximum number of update iterations
- `patience::Union{Int,Nothing}=nothing`: Number of consecutive identical states for convergence (default: 5)
- `miniterations_before_convergence::Union{Int,Nothing}=nothing`: Minimum iterations before checking convergence

### Returns
- `frames::Dict{Int64, Array{Int32,1}}`: Dictionary mapping iteration index to state vector
- `energydictionary::Dict{Int64, Float32}`: Dictionary mapping iteration index to energy
"""
function recover(model::MyClassicalHopfieldNetworkModel, sₒ::Array{Int32,1}, true_image_energy::Float32;
                 maxiterations::Int64=1000, patience::Union{Int,Nothing}=nothing,
                 miniterations_before_convergence::Union{Int,Nothing}=nothing)
    
    # get model parameters
    W = model.W
    b = model.b
    N = length(sₒ)
    
    # set defaults
    if patience === nothing
        patience = 5
    end
    if miniterations_before_convergence === nothing
        miniterations_before_convergence = patience
    end
    
    # initialize
    s = copy(sₒ)                                      # current state
    converged = false                                  # convergence flag
    t = 1                                             # iteration counter
    
    # output dictionaries
    frames = Dict{Int64, Array{Int32,1}}()
    energydictionary = Dict{Int64, Float32}()
    
    # state history queue for patience-based convergence
    state_history = CircularBuffer{Array{Int32,1}}(patience)
    
    # store initial state and energy
    frames[t] = copy(s)
    energydictionary[t] = energy(model, s)
    push!(state_history, copy(s))
    
    # main recovery loop
    while !converged
        t += 1
        
        # asynchronous update: choose a random neuron
        i = rand(1:N)
        
        # compute activation: h_i = Σⱼ w_ij * s_j - b_i
        h_i = sum(W[i, j] * s[j] for j in 1:N) - b[i]
        
        # update state using sign function
        if h_i >= 0
            s[i] = Int32(1)
        else
            s[i] = Int32(-1)
        end
        
        # store current state and energy
        frames[t] = copy(s)
        current_energy = energy(model, s)
        energydictionary[t] = current_energy
        
        # add to state history
        push!(state_history, copy(s))
        
        # check convergence (only after minimum iterations)
        if t >= miniterations_before_convergence
            
            # Check 1: State stability - all states in history are identical
            if length(state_history) == patience
                all_same = true
                first_state = state_history[1]
                for k in 2:patience
                    if hamming(state_history[k], first_state) != 0
                        all_same = false
                        break
                    end
                end
                if all_same
                    converged = true
                end
            end
            
            # Check 2: Energy at or below true minimum
            if current_energy <= true_image_energy
                converged = true
            end
        end
        
        # Check 3: Max iterations reached
        if t >= maxiterations
            converged = true
        end
    end
    
    return (frames, energydictionary)
end