# ================================================================
#  Haar Cascade 기반 얼굴 감지 → 흰 테두리 박스 + "사용자 인식" 표시
#
#  [실행 전 설치]
#  pip install opencv-contrib-python
#
#  → 웹캠 창이 열리며 'q' 키를 누르면 종료
# ================================================================

import cv2
import numpy as np


# ── Haar Cascade 얼굴 검출기 초기화 ─────────────────────────────
#
# [Haar Cascade 알고리즘]
# Viola & Jones (2001) 제안. Integral Image 기반으로 Haar-like Feature
# (밝기 차이 패턴)를 계산하고, AdaBoost로 학습된 약분류기 Cascade를 통과하면 얼굴로 판정.
# OpenCV 내장 XML 파일에 사전학습 가중치 포함 → 별도 학습 불필요.
# HOG+SVM 대비 근거리·상반신 환경에서 인식률이 훨씬 높음.

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    raise RuntimeError("Haar Cascade XML 로드 실패. OpenCV 설치를 확인하세요.")


# ── 함수 1: 얼굴 검출 ────────────────────────────────────────────
def detect_faces(frame: np.ndarray) -> list:
    """
    Haar Cascade를 사용하여 웹캠 프레임(BGR 이미지)에서 얼굴 Bounding Box를 검출한다.

    Args:
        frame: 웹캠에서 입력된 BGR numpy 배열

    Returns:
        faces: [(x, y, w, h), ...] 형태의 얼굴 위치 리스트

    파라미터:
        scaleFactor=1.1  : 이미지 축소 비율 (작을수록 정확하지만 느림)
        minNeighbors=5   : 검출 승인에 필요한 이웃 수 (클수록 오탐 감소)
        minSize=(30,30)  : 검출할 최소 얼굴 크기
    """
    # CLAHE: 조명이 고르지 않을 때 대비를 지역적으로 균일화
    # Haar Cascade는 밝기 차이(Haar Feature)에 민감하므로 이러한 전처리가 얼굴 검출 성능 향상에 도움이 된다.
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return list(faces) if len(faces) > 0 else []


# ── 함수 2: 시각화 ───────────────────────────────────────────────
def draw_detection(frame: np.ndarray, faces: list) -> np.ndarray:
    """
    검출된 얼굴 위치에 흰 테두리 박스와 "사용자 인식" 레이블 표시.

    Args:
        frame : BGR 프레임
        faces : [(x, y, w, h), ...] 얼굴 위치 목록

    Returns:
        frame : 시각화가 추가된 BGR 프레임
    """
    for (x, y, w, h) in faces:
        # 흰색 사각형 테두리 (두께 2px)
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      color=(255, 255, 255), thickness=2)

        # "사용자 인식" 텍스트: 박스 상단 10px 위
        # LINE_AA: 안티앨리어싱 적용으로 텍스트 선명하게 렌더링
        cv2.putText(
            frame,
            text="User Detected",
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # 우상단: 감지된 얼굴 수 표시
    cv2.putText(frame, f"Detected: {len(faces)}",
                (frame.shape[1] - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ── 함수 3: 메인 루프 ────────────────────────────────────────────
def run(camera_index: int = 0):
    """
    로컬 웹캠 실시간 얼굴 인식 루프.

    Args:
        camera_index: 웹캠 장치 번호 (내장 카메라=0, 외장 USB=1)

    종료: 'q' 키 입력
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"웹캠(인덱스 {camera_index})을 열 수 없습니다. "
            "카메라 연결 상태 또는 인덱스 번호를 확인하세요."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("웹캠 시작. 종료: 'q' 키")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            break

        # 거울 모드: 좌우 반전 → 사용자 움직임과 화면 방향 일치
        frame = cv2.flip(frame, 1)

        # 얼굴 검출 (Haar Cascade)
        faces = detect_faces(frame)

        # 시각화: 흰 박스 + 레이블
        frame = draw_detection(frame, faces)

        # 안내 텍스트 (좌하단)
        cv2.putText(frame, "Press 'q' to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료 완료.")


# ── 실행 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    run(camera_index=0)
