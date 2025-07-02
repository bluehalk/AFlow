# import os
# from groq import Groq

# client = Groq(
#     api_key="",
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
#     stream=False,
# )

# print(chat_completion.choices[0].message.content)

import numpy as np
import matplotlib.pyplot as plt

# Softmax function with temperature
def softmax_with_temperature(logits, temperature=1.0):
    """Compute softmax values for each class in logits, with a temperature scaling."""
    logits = np.array(logits)
    # Apply temperature scaling
    scaled_logits = logits / temperature
    exp_values = np.exp(scaled_logits)
    return exp_values / np.sum(exp_values)

# Example logits (e.g., output from a model)
logits = [2.0, 1.0, 0.1]

# Temperature values to test
temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]

# Plot the probability distributions for different temperatures
plt.figure(figsize=(10, 6))

for temp in temperatures:
    probs = softmax_with_temperature(logits, temperature=temp)
    plt.plot(probs, label=f"Temperature = {temp}")

plt.title("Softmax Output with Different Temperatures")
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()
