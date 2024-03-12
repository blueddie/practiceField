import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 텍스트 생성 함수 정의
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 텍스트 생성 예시
prompt = "나는?"
generated_text = generate_text(prompt, max_length=100)
print("Generated Text:", generated_text)