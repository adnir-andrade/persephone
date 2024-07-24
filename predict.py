import argparse
import json

from model_manager import ModelManager

def display_probabilities(probabilities):
    '''
    Formats a list of probabilities to be displayed as percentages.
    '''
        
    formatted_probs = [f"{prob * 100:.2f}%" for prob in probabilities]
    return formatted_probs

def main():
    '''
    Main function that predicts the class for an input image.
    It parses command-line arguments, loads the model from a checkpoint, 
    performs prediction, formats the probabilities, and then displays the class names 
    (only if a category mapping file is provided).
    '''

    parser = argparse.ArgumentParser(description="Welcome to Persephone! Here you can use your trained AI to see if it can guess what class a flower belongs.")
    parser.add_argument('image_path', type=str, help='Path to the input image. Example: ./flowers/manual_test/fox_glove.jpeg')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file. Example: ./checkpoints/checkpoint_vgg16_acc89_20240724.pth')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default: 5)')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names. Example: --category_names ./cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    model = ModelManager.load_checkpoint(file_name=args.checkpoint)
    probs, classes = model.predict(args.image_path, args.top_k, args.gpu)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        flower_names = [cat_to_name[str(flower)] for flower in classes]
    else:
        flower_names = classes

    print("Probabilities: ", display_probabilities(probs))
    print("Classes: ", flower_names)

if __name__ == '__main__':
    main()
