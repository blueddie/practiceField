from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 모델과 토크나이저 로드
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# 대화형 루프 시작
while True:
    # 한글 입력 받기
    korean_text = input("번역할 한글 문장을 입력하세요 (종료하려면 'exit' 입력): ")
    
    # 종료 조건 확인
    if korean_text.lower() == "exit":
        print("프로그램을 종료합니다.")
        break
    
    # 입력 문장을 토큰화하고 모델 입력 형식에 맞게 변환
    inputs = tokenizer(korean_text, return_tensors="pt", padding=True, truncation=True)

    # 번역 수행 (영어로 출력)
    translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])

    # 번역된 토큰을 텍스트로 디코딩하여 출력
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    print("번역 결과:", translated_text)