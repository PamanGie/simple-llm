from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_dataset_from_txt(file_path):
    """
    Function to load dataset from a text file.
    Each line in the text file is considered a separate sample.
    """
    # Read the text file with utf-8 encoding to avoid UnicodeDecodeError
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()  # Split lines to create dataset entries
    # Convert to Hugging Face dataset format
    return Dataset.from_dict({"text": data})

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenizes the input text for the model training.
    Sets input_ids and labels for language modeling.
    """
    # Tokenize text with padding and truncation
    tokenized = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    # Use input_ids as labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()  # Use clone() for tensor copying
    return tokenized

def train(model, tokenizer, train_file_path, output_dir, num_epochs=10, batch_size=8, learning_rate=2e-5):
    """
    Main training function that handles dataset loading, tokenization,
    and training of the model with specified arguments.
    """
    # Load dataset from txt file
    print(f"Loading dataset from {train_file_path}...")
    dataset = load_dataset_from_txt(train_file_path)
    print("Dataset loaded successfully.")

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])
    print("Tokenization completed.")

    # Data collator for language modeling (no mask, only causal language modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=200,
        push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,  # Replace with a separate validation split if available
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")
