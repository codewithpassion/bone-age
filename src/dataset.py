import os
from torch.utils.data import Dataset
from PIL import Image

class BoneAgeDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.transform = transform
        
        # Load labels from file
        self.labels = self._load_labels()
        
        # Get list of image filenames
        # self.image_filenames = os.listdir(data_dir)[:5000]
        self.image_filenames = os.listdir(data_dir)[5001:5200]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        # Get image filename and corresponding label
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.data_dir, image_filename)
        id = os.path.splitext(image_filename)[0]
        label = self.labels[id]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform is not None:
            image = self.transform(image)
        
        #print image name and label
        return image, label
    
    def _load_labels(self):
        labels = {}
        with open(self.labels_file, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                id, age, gender = line.strip().split(',')
                labels[id] = {'boneage': float(age), 'gender': int(bool(gender)), 'id': id}
        return labels