from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 모델과 토크나이저 로드
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 번역할 문장 입력
english_text = ""

# 토크나이저를 사용하여 입력 문장을 토큰화하고 모델에 입력 형식에 맞게 변환
inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)

# 번역 수행 (한국어로 출력)
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])

# 번역된 토큰을 텍스트로 디코딩하여 출력
translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
print("번역 결과:", translated_text)