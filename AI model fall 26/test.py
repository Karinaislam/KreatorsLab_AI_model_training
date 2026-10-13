# ============================================================
# TEST.PY - Talk to your TRAINED AI
# ============================================================
# This loads the AI that has learned from your data,
# and lets you ask it questions to see what it learned.
# ============================================================

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
TRAINED_BRAIN = "my-ai-trained-brain"

# ------------------------------------------------------------
# STEP 1: Load the base brain + the learning notebook you trained
# ------------------------------------------------------------
print("Loading your trained AI...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

have_trained_ai = os.path.exists(f"{TRAINED_BRAIN}/adapter_config.json")

if have_trained_ai:
    # PeftModel combines the base brain + your trained learning notebook
    model = PeftModel.from_pretrained(base_model, TRAINED_BRAIN)
    print("Loaded your TRAINED AI (base brain + your learning).")
else:
    print("WARNING: No trained AI found yet!")
    print(f"Missing folder: {TRAINED_BRAIN}")
    print("Please train your AI first (option [1] in the menu).")
    print()
    print("For now, this will show you the untrained base brain.")
    model = base_model

model.eval()
print()


# ------------------------------------------------------------
# STEP 2: The function that asks the AI a question
# ------------------------------------------------------------
def answer(question):
    """Feed a question to the AI and get back its answer + confidence."""
    # Format the question the same way we trained on
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Ask the AI to generate an answer (up to 60 tokens)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,              # Same question always gets same answer
            repetition_penalty=1.3,       # Discourage repeating itself
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract just the answer part (strip out the question prefix)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full.split("Answer:", 1)[-1].strip()

    # Calculate how confident the AI was, on average, in its answer
    with torch.no_grad():
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        logits = model(outputs).logits[0, inputs["input_ids"].shape[-1] - 1:-1]
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[range(len(gen_ids)), gen_ids]
        confidence = token_probs.mean().item() if len(gen_ids) > 0 else 0.0

    return reply, confidence


# ------------------------------------------------------------
# STEP 3: Interactive loop - ask as many questions as you want
# ------------------------------------------------------------
while True:
    print("=" * 50)
    print("         TEST YOUR TRAINED AI")
    print("=" * 50)
    print()

    question = input("Ask a question (or type 'back'): ").strip()

    if question.lower() == "back":
        break
    if not question:
        continue

    reply, confidence = answer(question)

    print()
    print("YOUR TRAINED AI SAYS:")
    print()
    print(f"  {reply}")
    print()
    print(f"  Confidence: {confidence:.0%}")
    print()

    input("Press Enter to try another...")
    print()

print()
print("Returning to menu...")