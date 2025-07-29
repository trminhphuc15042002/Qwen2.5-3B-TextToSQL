import os

# Cấu hình cache + offload để tránh lỗi
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
offload_dir = "./offload"
os.makedirs(offload_dir, exist_ok=True)

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load model PEFT
peft_model_id = "./adapter/Qwen2.5-3B-v2"
peft_config = PeftConfig.from_pretrained(peft_model_id)
base_model_id = peft_config.base_model_name_or_path

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    offload_folder=offload_dir
)

model = PeftModel.from_pretrained(
    base_model,
    peft_model_id,
    offload_folder=offload_dir
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# Hàm trả lời
def answer(context, question):
    prompt = f"### Context:\n{context.strip()}\n\n### Question:\n{question.strip()}\n\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply.split("### Answer:")[-1].strip()

# Lưu context toàn cục
state_context = gr.State("")

def chat_fn(history, context, question, context_state):
    if context.strip():
        context_state = context.strip()

    reply = answer(context_state, question)
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})

    return history, "", "", context_state

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Chat với GwenSQL - TEXT TO SQL")

    chatbot = gr.Chatbot(label="💬 GwenSQL", type="messages")
    context = gr.Textbox(label="📄 Context", lines=4, placeholder="Paste nội dung hoặc đoạn văn cần phân tích")
    question = gr.Textbox(label="❓ Question", lines=2, placeholder="Gõ câu hỏi...")
    context_state = gr.State("")

    send_btn = gr.Button("👉 Gửi câu hỏi")

    send_btn.click(chat_fn, [chatbot, context, question, context_state], [chatbot, question, context, context_state])
    question.submit(chat_fn, [chatbot, context, question, context_state], [chatbot, question, context, context_state])

demo.launch()
