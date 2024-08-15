import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore")

def generate_text(model_path, prompt, max_length=280, num_return_sequences=1):
    # Load the trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=60,
        temperature=0.7,
        do_sample=True,  # Ensure sampling mode is used for temperature, top_p, and top_k
        pad_token_id=tokenizer.eos_token_id  # Explicitly set the pad_token_id
    )

    # Decode the generated text and stop at the end of a sentence
    generated_texts = []
    for output_seq in output:
        text = tokenizer.decode(output_seq, skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        # Ensure the generated text ends at the end of a sentence
        end_of_sentence = text.find('.')
        if end_of_sentence != -1:
            text = text[:end_of_sentence+1]
        
        generated_texts.append(text)
    
    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}:\n{text}\n")

if __name__ == "__main__":
    model_path = "./trained_model"  # Path to the directory containing the trained model
    
    while True:
        # Prompt user for input
        prompt = input("What do you want to ask? ")
        
        # Check if user wants to quit
        if prompt.lower() == "quit":
            print("Exiting the chatbot. Goodbye!")
            break

        # Set generation parameters
        max_length = 280  # Maximum length of the generated text
        num_return_sequences = 1  # Number of generated text sequences

        # Generate text
        generate_text(model_path, prompt, max_length=max_length, num_return_sequences=num_return_sequences)
