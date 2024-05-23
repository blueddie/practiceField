import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

# Pre-trained R3D-18 모델 로드
model = r3d_18(pretrained=True)
model.eval()

# 영상 데이터 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    return frame

def detect_anomalies(frames):
    frames = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)  # (채널, 시간, 높이, 너비)로 변환
    with torch.no_grad():
        predictions = model(frames)
    return predictions

# 웹캠 피드
cap = cv2.VideoCapture(0)

frame_sequence = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = preprocess_frame(frame)
    frame_sequence.append(processed_frame)
    
    if len(frame_sequence) == 16:  # 예시로 16 프레임씩 처리
        predictions = detect_anomalies(frame_sequence)
        # 예측 결과에 따른 알림 생성 (생략)
        frame_sequence = []

    # 실시간 영상 출력
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
