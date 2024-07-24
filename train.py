import argparse

from model_manager import ModelManager
from loader_manager import LoaderManager

def main():
    '''
    Main function to set up and run the training and evaluation of a model.
    It parses command line arguments, initializes the model and data loaders,
    trains the model, tests it, and then finally saves the trained model to a checkpoint.
    '''

    parser = argparse.ArgumentParser(description="Welcome to Persephone! Here you can train a new neural network on a dataset and saves the model as a checkpoint.")
    parser.add_argument('data_dir', type=str, help='Directory with training data')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (default: vgg16). Options: vgg13 | vgg16 | vgg19 | densenet121 | densenet201')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=0.00009, help='Learning rate (default: 0.00009)')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units (default: 1024)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints (default: ./checkpoints)')

    args = parser.parse_args()

    model = ModelManager(
        arch = args.arch, 
        epochs = args.epochs, 
        learning_rate = args.learning_rate, 
        h1_size = args.hidden_units,
        use_gpu = args.gpu,
        )
    
    loaders = LoaderManager(args.data_dir)

    model.start_training(loaders)
    model.test_model(loaders.test_loader)
    model.save_checkpoint(args.save_dir)

if __name__ == '__main__':
    main()