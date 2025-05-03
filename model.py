import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

with open("text.txt", "r", encoding="utf-8") as f:
    conversation = f.read()

prompt = (
    "Я отправляю тебе переписку в Telegram. "
    "Сделай, пожалуйста, краткое содержание на русском языке, обязательно упомяни авторов сообщений.\n\n"
    + conversation
)

messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

output_ids = output[0][inputs.input_ids.shape[1]:].tolist()

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

summary = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

print("\n[SUMMARY]:\n")
for line in summary.split('. '):
    print(line.strip() + '.')
