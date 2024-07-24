from torch import utils
from torchvision import datasets, transforms

class LoaderManager:
    '''
    Manages data loading and preprocessing for training, validation, and testing datasets.

    Attributes:
        train_data (datasets.ImageFolder): Training data with transformations applied.
        train_loader (utils.data.DataLoader): DataLoader for the training data.
        valid_loader (utils.data.DataLoader): DataLoader for the validation data.
        test_loader (utils.data.DataLoader): DataLoader for the test data.
        data_dir (str): Directory containing the training data.
    '''

    def __init__(self, data_dir):
        self.train_data = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.data_dir = data_dir
        self._initialize_loader()

    def _initialize_loader(self):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        self.train_data = datasets.ImageFolder(self.data_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder('./flowers/valid', transform=valid_transforms)
        test_data = datasets.ImageFolder('./flowers/test', transform=test_transforms)

        self.train_loader = utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.valid_loader = utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
        self.test_loader = utils.data.DataLoader(test_data, batch_size=64)