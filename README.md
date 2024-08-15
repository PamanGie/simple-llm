# Simple-LLM
Simple LLM with GPT 2 and Transformers

# Sampling
This LLM uses nucleus sampling (top-p sampling) combined with top-k sampling and temperature-based sampling.

# How to Use?
1. Prepare your dataset in .txt format
2. Open main.py and change your dataset.txt file
3. run ``python main.py`` and wait until the training done
4. The duration of training is depends on capability of your GPU card
5. You can play the parameters to find best result of your LLM
6. after training is done, a new folder "trained_model" will appear with your token
7. next, you can run ``python generate.py`` and ask anything that related on your dataset

# Required Library
1. Anaconda
2. transformers ``pip install transformers``
3. torch ``pip install torch``

# Required Hardware
1. Minimum GTX 1650 Super 4Gb or RTX 3060 12 Gb for Better Speed of Training
2. Intel Core i5 Gen 8 or Latel Intel Core i7 for better performance
3. 8Gb Ram or 32 Gb Ram

# Result
![Alt text](https://raw.githubusercontent.com/PamanGie/simple-llm/main/llm.PNG)
