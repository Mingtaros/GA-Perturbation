import torch

# get accuracy score when image is perturbed
def perturb(test_dataset, perturbation):
    perturbed_images = []
    perturbed_labels = []

    for data, target in test_dataset:
        # Add perturbation to each image
        perturbed_data = data + torch.tensor(perturbation, dtype=torch.float32)
        perturbed_images.append(perturbed_data)
        perturbed_labels.append(target)

    # Create a new dataset with perturbed images
    perturbed_dataset = torch.utils.data.TensorDataset(torch.stack(perturbed_images), torch.tensor(perturbed_labels))

    return perturbed_dataset