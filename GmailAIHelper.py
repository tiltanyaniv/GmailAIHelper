from gpt4all import GPT4All

# Load the Llama 3 8B Instruct model
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Generate output
output = model.generate("Answer this prompt by saying Hello LLM")

# Print the output
print(output)