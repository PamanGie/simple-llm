from model import load_model
from training import train

def main():
    model_name = "gpt2"  # Change if you want to use a different model
    train_file_path = "dataset2.txt"  # Change to the path of your dataset
    output_dir = "./trained_model"  # Directory to save the trained model and tokenizer

    # Load model and tokenizer
    model, tokenizer = load_model(model_name)

    # Set training parameters
    num_epochs = 10  # Number of epochs
    batch_size = 8   # Batch size
    learning_rate = 2e-5  # Learning rate

    # Train model
    train(model, tokenizer, train_file_path, output_dir, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

if __name__ == "__main__":
    main()
