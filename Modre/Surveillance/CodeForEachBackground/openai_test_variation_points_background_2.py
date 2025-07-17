

import openai
import json
import pandas as pd

# Your OpenAI API key
#openai.api_key = ''

# Step 1: Load the goal model from a txt file
with open('ObjectDetectorGoalModel.txt', 'r') as file:
    goal_model = json.load(file)  # Load the JSON from the txt file

# Step 2: Load background information from a file
with open('background_information_2.txt', 'r') as file:
    raw_background = file.read()

# Convert goal model JSON back to string format
goal_model_str = json.dumps(goal_model)

# Step 3: Store results for comparison
variation_points_with_rationale = []

# Combine background information with the goal model and with the prompt task. 
prompt = (f"Background: {raw_background}\n\nHere is a goal model: {goal_model_str}."
          " Detect independent explicit designed variation points in each classifier agent in"
          "this goal model.")


# Call the OpenAI API to detect variation points
response = openai.chat.completions.create(
    model="gpt-3.5-turbo-0125" , 
    messages=[
        {"role": "system", "content": "You are a software engineer."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=400,
    temperature=0.2
)

# Get the response text
variation_text = response.choices[0].message.content

# Store the result for this background
variation_points_with_rationale.append({
    "Detected Variation Points and Rationale": variation_text
})


# Step 5: Save the response text in a .txt file
with open('variation_points_background_2.txt', 'w') as txt_file:
    txt_file.write(variation_text)