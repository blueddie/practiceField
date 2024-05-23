import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18

# Pre-trained R3D-18 모델 로드
model = r3d_18(pretrained=True)
model.eval()

# UCF101 데이터셋의 예시 클래스 레이블
CLASS_LABELS = [
    "Applying Eye Makeup", "Archery", "Baby Crawling", "Balance Beam", "Band Marching", 
    "Baseball Pitch", "Basketball", "Basketball Dunk", "Bench Press", "Biking"
    # 실제 사용 시, 모든 101개의 클래스 레이블을 여기에 추가해야 합니다.
]

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

def get_action_label(predictions):
    # 각 클래스에 대한 예측 확률을 구함
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    # 가장 높은 확률을 가진 클래스의 인덱스를 구함
    predicted_class = torch.argmax(probabilities, dim=1).item()
    # 예측된 클래스의 레이블을 반환
    return CLASS_LABELS[predicted_class]

# 동영상 파일 경로 설정
video_path = r'C:\Users\ed\Downloads\Sample\Sample\01.원천데이터\03.이상행동\12.절도\C_3_12_1_BU_SMC_08-07_13-30-11_CC_RGB_DF2_M1.mp4'
cap = cv2.VideoCapture(video_path)

frame_sequence = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = preprocess_frame(frame)
    frame_sequence.append(processed_frame)
    
    if len(frame_sequence) == 16:  # 예시로 16 프레임씩 처리
        predictions = detect_anomalies(frame_sequence)
        action_label = get_action_label(predictions)
        print(f"The person is performing: {action_label}")
        frame_sequence = []

    # 실시간 영상 출력
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
