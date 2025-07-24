using Images
using FileIO
using Random
using BSON  # For saving data in a binary format

# Function to load and preprocess images
function load_images_from_directory(directory::String, label::Int, image_size::Tuple{Int, Int})
    images = []
    labels = []
    for (root, dirs, files) in walkdir(directory)
        for file in files
            if endswith(file, ".jpg")
                img_path = joinpath(root, file)
                img = load(img_path)
                img_resized = imresize(img, image_size)  # Resize to 32x32
                push!(images, Float32.(channelview(img_resized)))  # Convert to Float32 array
                push!(labels, label)
            end
        end
    end
    return images, labels
end

# Main function to create training and test sets
function create_dataset(input_dirs::Vector{String}, output_dir::String, image_size::Tuple{Int, Int}=(32, 32), train_split::Float64=0.8)
    all_images = []
    all_labels = []
    
    # Load images from each directory
    for (i, dir) in enumerate(input_dirs)
        println("Processing directory: $dir")
        images, labels = load_images_from_directory(dir, i, image_size)
        append!(all_images, images)
        append!(all_labels, labels)
    end
    
    # Shuffle and split into training and test sets
    indices = shuffle(1:length(all_images))
    split_idx = Int(floor(train_split * length(indices)))
    train_indices = indices[1:split_idx]
    test_indices = indices[split_idx+1:end]
    
    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]
    
    # Save datasets
    println("Saving datasets...")
    mkpath(output_dir)
    #BSON.@save joinpath(output_dir, "train.bson") train_images train_labels
    BSON.@save joinpath(output_dir, "test.bson") test_images test_labels
    println("Datasets saved successfully!")
end

# Example usage
input_dirs = ["data/test/FAKE", "data/test/REAL", "data/train/FAKE", "data/train/REAL"]  # Add paths to your directories
output_dir = "data.csv"
create_dataset(input_dirs, output_dir)