import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from models import InceptionV3_CBAM, Xception_ResNet50
from dataset import BoneAgeDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
num_epochs = 50
batch_size = 32
learning_rate = 0.001

# Define paths to data and labels
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
data_dir = os.path.join(root_path, "data")
train_data = os.path.join(data_dir, "training")
train_labels = os.path.join(data_dir, "boneage-training-dataset.csv")

# Create a directory to store the snapshots
snapshot_dir = os.path.join(data_dir, "snapshots")
os.makedirs(snapshot_dir, exist_ok=True)

# Define data transforms and data loaders
train_transform = transforms.Compose([
    # transforms.Resize((200, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BoneAgeDataset(train_data, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

# Define your models
region_model = InceptionV3_CBAM(num_classes=2).to(device)
age_model = Xception_ResNet50().to(device)

# Define loss function and optimizer
region_criterion = nn.CrossEntropyLoss()
age_criterion = nn.MSELoss()
region_optimizer = optim.Adam(region_model.parameters(), lr=learning_rate)
age_optimizer = optim.Adam(age_model.parameters(), lr=learning_rate)

start_epoch = 0

# Load the model snapshots if they exist
# search for the latest snapshot
latest_snapshot = None
for file in os.listdir(snapshot_dir):
    if file.startswith("snapshot_epoch_") and file.endswith(".pth"):
        epoch = int(file.split("_")[-1].split(".")[0])
        if latest_snapshot is None or epoch > latest_snapshot:
            start_epoch = epoch
            
            print(f"Found snapshot at epoch {epoch} in file {file}")
            checkpoint = torch.load(os.path.join(snapshot_dir, file))
            start_epoch = checkpoint['epoch']
            region_model.load_state_dict(checkpoint['region_model_state_dict'])
            age_model.load_state_dict(checkpoint['age_model_state_dict'])
            region_optimizer.load_state_dict(checkpoint['region_optimizer_state_dict'])
            age_optimizer.load_state_dict(checkpoint['age_optimizer_state_dict'])            
            
            print(f"Loaded model snapshot at epoch {start_epoch} from file {latest_snapshot}")


# Training loop
for epoch in range(start_epoch, num_epochs):
    batch_number = 0
    for images, labels in train_loader:
        print(f"Batch {batch_number+1}/{len(train_loader)} - Epoch {epoch+1}/{num_epochs}")
        batch_number += 1
        images = images.to(device)
        ages = labels['boneage'].unsqueeze(1).to(device).float()
        genders = labels['gender'].to(device)

        # Train the region extraction model
        region_optimizer.zero_grad()
        region_outputs = region_model(images)
        region_loss = region_criterion(region_outputs, genders)
        region_loss.backward()
        region_optimizer.step()

        # Extract carpal and metacarpal regions using the trained region model
        with torch.no_grad():
            _, predicted_regions = torch.max(region_outputs, 1)
            carpal_mask = predicted_regions == 0
            metacarpal_mask = predicted_regions == 1
            carpal_regions = images[carpal_mask]
            metacarpal_regions = images[metacarpal_mask]

        # Train the bone age estimation model
        age_optimizer.zero_grad()
        combined_regions = torch.cat((carpal_regions, metacarpal_regions), dim=0)
        age_outputs = age_model(combined_regions)
        age_loss = age_criterion(age_outputs, ages)
        age_loss.backward()
        age_optimizer.step()

    # Print the losses for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Region Loss: {region_loss.item():.4f}, Age Loss: {age_loss.item():.4f}")
    
    # Save the model snapshots at the end of each epoch
    snapshot_path = os.path.join(snapshot_dir, f'snapshot_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'region_model_state_dict': region_model.state_dict(),
        'age_model_state_dict': age_model.state_dict(),
        'region_optimizer_state_dict': region_optimizer.state_dict(),
        'age_optimizer_state_dict': age_optimizer.state_dict(),
        'region_loss': region_loss.item(),
        'age_loss': age_loss.item()
    }, snapshot_path)
    # delete all except the latest snapshot
    for file in os.listdir(snapshot_dir):
        if file.startswith("snapshot_epoch_") and file.endswith(".pth") and file != f'snapshot_epoch_{epoch+1}.pth':
            os.remove(os.path.join(snapshot_dir, file))    
    print(f"Model snapshot saved at {snapshot_path}")
    

# Save the trained models
torch.save(region_model.state_dict(), 'region_model.pth')
torch.save(age_model.state_dict(), 'age_model.pth')