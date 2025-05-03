import os
from dotenv import load_dotenv
from collections import deque
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Init model
MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# Message buffer
message_buffer = deque(maxlen=500)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        author = update.message.from_user.first_name or "Unknown"
        text = update.message.text
        message_buffer.append(f"{author}: {text}")
        print(f"[LOGGED] {author}: {text}")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /summ N — где N количество последних сообщений.")
        return

    n = int(context.args[0])
    if n > len(message_buffer):
        await update.message.reply_text(f"Доступно только {len(message_buffer)} сообщений.")
        return

    print(f"[SUMMARIZE] Summarizing last {n} messages...")
    messages = list(message_buffer)[-n:]
    full_text = "\n".join(messages)

    prompt = (
        "Я отправляю тебе переписку в Telegram. "
        "Сделай, пожалуйста, краткое содержание на русском языке, обязательно упомяни авторов сообщений.\n\n"
        + full_text
    )

    chat_messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    output_ids = output[0][model_inputs["input_ids"].shape[1]:].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    summary = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    await update.message.reply_text(f"Сводка:\n{summary}")

def main():
    print("[START] Bot is starting...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("summ", summarize_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
