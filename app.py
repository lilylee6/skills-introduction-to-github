from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model (dùng GPT-2 để chạy nhanh)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_story(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def home():
    story = ""
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        story = generate_story(prompt)
    return render_template("index.html", story=story)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)