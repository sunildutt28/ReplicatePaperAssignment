using Flux, Statistics, Random, MLUtils, Images, FileIO, ProgressMeter, Optimisers
using Flux: onehotbatch, onecold
using Printf

Random.seed!(42)  # Set random seed for reproducibility

# 1. Safe LeNet-5 with output clamping
function OriginalLeNet5(input_channels=1, output_size=10)
    Chain(
        Conv((5,5), input_channels=>6, tanh),
        MeanPool((2,2), stride=(2,2)),
        Conv((5,5), 6=>16, tanh),
        MeanPool((2,2), stride=(2,2)),
        Flux.flatten,
        Dense(400=>120, tanh),
        Dense(120=>84, tanh),
        Dense(84=>output_size),
        x -> clamp.(x, -20, 20),  # Prevent numerical instability
        softmax  # Now safe with clamped inputs
    )
end

# 2. Safe cross-entropy loss
function safe_crossentropy(ŷ, y)
    # Clamp predictions to avoid log(0)
    ϵ = eps(Float32)
    ŷ = clamp.(ŷ, ϵ, 1-ϵ)
    -sum(y .* log.(ŷ)) / size(y, 2)
end

# 3. Robust image loading
function load_image(path)
    try
        img = load(path)
        gray_img = Gray.(img)
        resized = imresize(gray_img, (32, 32))
        img_array = Float32.(channelview(resized))
        normalized = (img_array .- 0.1307) ./ 0.3081
        return reshape(normalized, 32, 32, 1)
    catch e
        @warn "Failed to process $path: $e"
        return nothing
    end
end

# 4. Dataset validation
function load_dataset(base_path)
    categories = sort([d for d in readdir(base_path) if isdir(joinpath(base_path, d))])
    image_arrays = []
    labels = []
    
    @showprogress for (label_idx, category) in enumerate(categories)
        healthy_path = joinpath(base_path, category, "healthy")
        isdir(healthy_path) || continue
        
        for file in readdir(healthy_path)
            if endswith(lowercase(file), ".jpg") || endswith(lowercase(file), ".png")
                img_array = load_image(joinpath(healthy_path, file))
                if !isnothing(img_array)
                    push!(image_arrays, img_array)
                    push!(labels, label_idx)
                end
            end
        end
    end
    
    # Verify we have enough samples
    length(image_arrays) > 100 || error("Insufficient samples ($(length(image_arrays))")
    
    X = cat(image_arrays..., dims=4)
    y = onehotbatch(labels, 1:length(categories))
    return (X, y), categories
end

# 5. Training with safeguards
function train_model(model, train_loader, val_loader; epochs=10)
    opt = Optimisers.Adam(0.001)
    state = Optimisers.setup(opt, model)
    best_acc = 0.0
    best_model = deepcopy(model)
    
    for epoch in 1:epochs
        # Training
        for (x, y) in train_loader
            grads = gradient(model) do m
                safe_crossentropy(m(x), y)
            end
            state, model = Optimisers.update(state, model, grads[1])
        end
        
        # Validation
        acc = try
            accuracy(model, val_loader)
        catch e
            @warn "Validation failed: $e"
            0.0
        end
        
        @info @sprintf("Epoch %2d: Val Acc = %.2f%%", epoch, acc*100)
        
        # Early stopping
        if acc > best_acc
            best_acc = acc
            best_model = deepcopy(model)
        elseif epoch > 5 && acc < 0.5*best_acc
            @warn "Early stopping at epoch $epoch"
            break
        end
    end
    
    return best_model
end

# 6. Main training pipeline
function train_and_evaluate(dataset_path; k=10, epochs=30, batchsize=32)
    try
        (X, y), categories = load_dataset(dataset_path)
        n_samples = size(X, 4)
        @info "Training on $n_samples samples ($(length(categories)) classes)"
        
        # Create simpler folds for stability
        folds = collect(Iterators.partition(shuffle(1:n_samples), ceil(Int, n_samples/k)))
        k = length(folds)
        accuracies = zeros(k)
        
        @showprogress for fold in 1:k
            test_idx = folds[fold]
            train_idx = vcat(folds[setdiff(1:k, fold)]...)
            
            # Simple 80/20 split
            split_idx = floor(Int, 0.8*length(train_idx))
            train_loader = DataLoader(
                (X[:,:,:,train_idx[1:split_idx]], y[:,train_idx[1:split_idx]]),
                batchsize=batchsize, shuffle=true
            )
            val_loader = DataLoader(
                (X[:,:,:,train_idx[split_idx+1:end]], y[:,train_idx[split_idx+1:end]]),
                batchsize=batchsize
            )
            test_loader = DataLoader(
                (X[:,:,:,test_idx], y[:,test_idx]),
                batchsize=batchsize
            )
            
            model = OriginalLeNet5(1, length(categories))
            model = train_model(model, train_loader, val_loader, epochs=epochs)
            
            acc = accuracy(model, test_loader)
            accuracies[fold] = acc
            @info @sprintf("Fold %d: Test Acc = %.2f%%", fold, acc*100)
        end
        
        # Filter successful folds
        valid_accs = accuracies[accuracies .> 0]
        if !isempty(valid_accs)
            @info "\nFinal Results (from $(length(valid_accs)) folds):"
            @info @sprintf("Mean Accuracy: %.2f%%", mean(valid_accs)*100)
            @info @sprintf("Std. Deviation: %.2f%%", std(valid_accs)*100)
        else
            @warn "All folds failed!"
        end
        
        return accuracies
    catch e
        @error "Pipeline failed" exception=(e, catch_backtrace())
        return zeros(k)
    end
end

# Helper functions
function accuracy(model, loader)
    correct = 0
    total = 0
    for (x, y) in loader
        ŷ = model(x)
        correct += sum(onecold(ŷ) .== onecold(y))
        total += size(y, 2)
    end
    return correct / total
end

# Run with reduced parameters initially
dataset_path = "C:\\Assignments\\DeepLearning\\ReplicatePaperAssignment\\hb74ynkjcn-5"
results = train_and_evaluate(dataset_path, k=10, epochs=30, batchsize=32) 