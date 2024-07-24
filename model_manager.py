from datetime import datetime
from math import floor
from os import path

import torch
from torch import nn, optim
from torchvision import models

from classifier import Classifier
from loader_manager import LoaderManager
from utils import process_image


class ModelManager:
    criterion = nn.NLLLoss()

    def __init__(self, arch, epochs, learning_rate, h1_size, use_gpu, input_size=25088, output_size=102):
        '''
        Initializes the ModelManager class with architecture, training parameters, and device configuration.
        '''

        self.input_size = input_size
        self.arch = arch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.h1_size = h1_size
        self.output_size = output_size
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model_accuracy = 0
        self.checkpoint = {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        '''
        Initializes the model architecture and prepares it for training.
        '''

        print("Creating model...")
        model = self.set_arch(self.arch)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = Classifier(self.input_size, self.h1_size, self.output_size)
        model.to(self.device)

        if self.use_gpu: torch.cuda.empty_cache()

        print("... Model created!\n")
        return model
    
    def start_training(self, loaders: LoaderManager):
        '''
        Train the model using the provided data loaders.
        Calculates and prints training and validation losses.
        '''

        print("Starting training...")
        optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
        steps = 0
        print_every = 5
        train_losses, validation_losses = [], []


        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0
            for inputs, labels in loaders.train_loader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                log_probability = self.model(inputs)
                loss = self.criterion(log_probability, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    self.model.eval()
                    
                    with torch.no_grad():
                        for inputs, labels in loaders.valid_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model(inputs)
                            batch_loss = self.criterion(logps, labels)
                            
                            validation_loss += batch_loss.item()
                            
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                    train_losses.append(running_loss/len(loaders.train_loader))
                    validation_losses.append(validation_loss/len(loaders.valid_loader))
                            
                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {validation_loss/len(loaders.valid_loader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(loaders.valid_loader) * 100:.3f} %")
                    running_loss = 0
                    self.model.train()
        
        self.model.class_to_idx = loaders.train_data.class_to_idx
        self.set_checkpoint(optimizer)
        print("... Training complete!\n")

    def test_model(self, loader):
        '''
        Tests the model and prints the test accuracy.
        '''

        print("Testing model...")
        accuracy = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model(inputs)
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        self.model_accuracy = accuracy/len(loader) * 100
        acc = accuracy/len(loader) * 100
        print(f"Test accuracy: {acc:.3f} %")

        print("... Test complete!\n")

    def predict(self, image_path, topk = 5, use_gpu = False):
        '''
        Makes a prediction of a given image and returns the top K probabilities and class labels.
        '''

        print("Persephone is checking your flower...")

        image = process_image(image_path)
        image_input = torch.from_numpy(image).float()
        image_input = image_input.unsqueeze(0)
        self.device = torch.device("cuda" if use_gpu else "cpu")
        image_input = image_input.to(next(self.model.parameters()).device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_input)
            probabilities = torch.exp(output)
            
            top_p, top_class = probabilities.topk(topk, dim=1)
            
            top_p = top_p.squeeze().cpu().numpy()
            top_class = top_class.squeeze().cpu().numpy()
            
            idx_to_class = {value: key for key, value in self.model.class_to_idx.items()}
            top_classes = [idx_to_class[idx] for idx in top_class]
        
        print("... Prediction complete!\n")
        return top_p, top_classes

    def set_arch(self, arch):
        '''
        Sets the model architecture based on the specified architecture type.
        '''

        if arch == 'vgg13': return models.vgg16(weights='DEFAULT')
        if arch == 'vgg16': return models.vgg16(weights='DEFAULT')
        if arch == 'vgg19': return models.vgg19(weights='DEFAULT')
        if arch == 'densenet121':
            self.input_size = 1024
            return models.densenet121(weights='DEFAULT')
        if arch == 'densenet201':
            self.input_size = 1024
            return models.densenet121(weights='DEFAULT')

        raise ValueError(f"Architecture {arch} is not supported. Using vgg16.")

    def set_checkpoint(self, optimizer):
        '''
        Sets the model checkpoint dictionary with current model state and optimizer state.
        '''

        self.checkpoint = {
            'arch': self.arch,
            'input_size': self.input_size,
            'h1_size': self.h1_size,
            'output_size': self.output_size,
            'hidden_layers': [each.out_features for each in self.model.classifier.children() if isinstance(each, nn.Linear)],
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': self.model.class_to_idx,
        }

    def save_checkpoint(self, save_dir):
        '''
        Saves the model checkpoint to a file.
        The file name will consist of architecture, model accuracy and a timestamp.
        '''

        print("Saving model...")
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'checkpoint_{self.arch}_acc{floor(self.model_accuracy)}_{timestamp}.pth'
        filepath = path.join(save_dir, filename)

        torch.save(self.checkpoint, filepath)
        print("... Saved successfully!\n")

    @classmethod
    def load_checkpoint(cls, file_name):
        '''
        Loads a model checkpoint from a file and returns an instance of ModelManager
        with the loaded model.
        '''

        print("Loading model...\n")
        checkpoint = torch.load(file_name)
        
        instance = cls(
            arch=checkpoint['arch'],
            epochs=1,
            learning_rate=0.00009,
            h1_size=checkpoint['h1_size'],
            use_gpu=torch.cuda.is_available(),
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size']
        )

        model = cls.set_arch(cls, arch = checkpoint['arch'])
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = Classifier(
            input_size=checkpoint['input_size'],
            h1_size=checkpoint['h1_size'],
            output_size=checkpoint['output_size']
        )
        
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
        instance.model = model
        
        print("... Loaded successfully!\n")
        return instance

