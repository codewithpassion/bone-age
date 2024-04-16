import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import InceptionV3_CBAM, Xception_ResNet50
from dataset import BoneAgeDataset

# Define paths to data and labels
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
data_dir = os.path.join(root_path, "data")
test_data = os.path.join(data_dir, "test")
test_labels = os.path.join(data_dir, "boneage-test-dataset.csv")
model_dir = os.path.join(data_dir, "5000")

train_data = os.path.join(data_dir, "training")
train_labels = os.path.join(data_dir, "boneage-training-dataset.csv")


# Set the paths to the test data and labels
# test_data_dir = 'path/to/test/data'
# test_labels_file = 'path/to/test/labels.csv'

# Set the path to the trained models
region_model_path = os.path.join(model_dir,'region_model.pth')
age_model_path = os.path.join(model_dir, 'age_model.pth')

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data transforms
test_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the test dataset and data loader
# test_dataset = BoneAgeDataset(test_data, test_labels, test_transform)
test_dataset = BoneAgeDataset(train_data, train_labels, test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained models
region_model = InceptionV3_CBAM(num_classes=2)
region_model.load_state_dict(torch.load(region_model_path))
region_model.to(device)
region_model.eval()

age_model = Xception_ResNet50()
age_model.load_state_dict(torch.load(age_model_path))
age_model.to(device)
age_model.eval()

# Iterate over the test data
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        
        # Extract carpal and metacarpal regions using the trained region model
        region_outputs = region_model(images)
        _, predicted_regions = torch.max(region_outputs, 1)
        carpal_mask = predicted_regions == 0
        metacarpal_mask = predicted_regions == 1
        carpal_regions = images[carpal_mask]
        metacarpal_regions = images[metacarpal_mask]
        
        # Estimate the bone age using the trained age model
        combined_regions = torch.cat((carpal_regions, metacarpal_regions), dim=0)
        age_outputs = age_model(combined_regions)
        
        # Get the image name and estimated bone age
        image_name = labels['id'][0]
        age = labels['boneage'][0]
        estimated_age = age_outputs.item()
        
        # Output the image name and estimated bone age
        print(f"Image: {image_name}, Estimated Bone Age: {estimated_age:.2f} months - actual age: {age:.2f} months - delta: {abs(estimated_age - age):.2f} months")