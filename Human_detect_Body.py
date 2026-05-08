# ================================================================
#  드론 제어 — 얼굴 인식 → 신체 자세 제스처 제어
#
#  [방식]
#  - 얼굴: OpenCV Haar Cascade (3초 인식)
#  - 신체: YOLOv8-pose → 어깨/손목/머리/엉덩이 키포인트 추출
#
#  [상태 흐름]
#  FACE → (3초) → WAIT_POSE → (자세 감지) → GESTURE
#
#  
# ================================================================

import cv2
import numpy as np
import time
from ultralytics import YOLO

# ── 상수 ─────────────────────────────────────────────────────────
FACE_DWELL_SEC   = 3.0   # 얼굴 인식 유지 시간(초)
KP_CONF          = 0.4   # 키포인트 최소 신뢰도
TPOSE_Y_THR      = 0.10  # T자 판정: 손목-어깨 y 차이 임계값 (정규화)
CHEST_THR        = 0.14  # 가슴 근접 판정 임계값 (정규화)

# 상태
STATE_FACE      = 'FACE'
STATE_WAIT_POSE = 'WAIT_POSE'
STATE_GESTURE   = 'GESTURE'

# 제스처 색상 (BGR)
COLOR_TAKEOFF  = (180,   0, 180)   # 보라  – Takeoff/Hover
COLOR_ROLL     = (0,     0, 255)   # 빨강  – Roll
COLOR_THROTTLE = (0,   165, 255)   # 주황  – Throttle
COLOR_YAW      = (0,   200,   0)   # 초록  – Yaw
COLOR_LANDING  = (0,   220, 220)   # 시안  – Landing
COLOR_WHITE    = (255, 255, 255)
COLOR_GRAY     = (160, 160, 160)

# COCO 키포인트 인덱스
KP_NOSE     = 0
KP_L_SHLDR  = 5
KP_R_SHLDR  = 6
KP_L_WRIST  = 9
KP_R_WRIST  = 10
KP_L_HIP    = 11
KP_R_HIP    = 12

# ── 검출기 초기화 ─────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("[로딩] YOLOv8-pose 모델 초기화 중...")
pose_model = YOLO('yolov8n-pose.pt')   # 최초 실행 시 자동 다운로드
print("[로딩] 완료.")



# ────────얼굴 검출 ────────────────────────────────────────────
def detect_faces(frame: np.ndarray) -> list:
    """Haar Cascade + CLAHE 전처리로 얼굴 BBox 검출."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    faces   = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return list(faces) if len(faces) > 0 else []

# ────────신체 키포인트 추출 ──────────────────────────────────
def get_keypoints(frame: np.ndarray) -> dict | None:
    """
    YOLOv8-pose로 가장 큰 인물의 키포인트 추출.
    Returns: {'nose':(x,y,c), 'l_shldr':..., ...} 또는 None
    """
    results = pose_model(frame, verbose=False)
    if not results or results[0].keypoints is None:
        return None

    kps_data = results[0].keypoints.data  # (N, 17, 3)
    if kps_data.shape[0] == 0:
        return None

    # 화면에서 가장 큰 사람 선택
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 1:
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best  = int(areas.argmax())
    else:
        best = 0

    kps = kps_data[best].cpu().numpy()   # (17, 3) → [x, y, conf]

    def g(i):
        return float(kps[i, 0]), float(kps[i, 1]), float(kps[i, 2])

    return {
        'nose':    g(KP_NOSE),
        'l_shldr': g(KP_L_SHLDR),
        'r_shldr': g(KP_R_SHLDR),
        'l_wrist': g(KP_L_WRIST),
        'r_wrist': g(KP_R_WRIST),
        'l_hip':   g(KP_L_HIP),
        'r_hip':   g(KP_R_HIP),
    }

# ──────── 제스처 분류 ──────────────────────────────────────────
def classify_gesture(kps: dict, fw: int, fh: int) -> dict | None:
    """
    신체 키포인트로 제스처 분류. 우선순위:
    1. Landing      — 양쪽 손목 모두 가슴 근처
    2. Throttle(+)  — 양쪽 손목 모두 머리(코) 위
    3. Throttle(-)  — 양쪽 손목 모두 엉덩이 아래
    4. Yaw(-/+)     — 한쪽 손목 가슴 근처
    5. Roll(+/-)    — 한쪽 팔만 어깨보다 위로
    6. Takeoff/Hover— 양팔 T자 (손목 y ≈ 어깨 y)
    직관적인 인식을 위해 pitch 가 생략되었으나 중간 발표 이후 추가 예정
    """
    # 신뢰도 낮은 키포인트가 있으면 판단 불가
    for v in kps.values():
        if v[2] < KP_CONF:
            return None

    # 정규화 좌표 추출
    def nx(x): return x / fw
    def ny(y): return y / fh

    nose_y  = ny(kps['nose'][1])
    ls_x, ls_y = nx(kps['l_shldr'][0]), ny(kps['l_shldr'][1])
    rs_x, rs_y = nx(kps['r_shldr'][0]), ny(kps['r_shldr'][1])
    lw_x, lw_y = nx(kps['l_wrist'][0]), ny(kps['l_wrist'][1])
    rw_x, rw_y = nx(kps['r_wrist'][0]), ny(kps['r_wrist'][1])
    lh_y = ny(kps['l_hip'][1])
    rh_y = ny(kps['r_hip'][1])
    hip_y = (lh_y + rh_y) / 2

    # 가슴 중심 추정 (어깨 중간, 약간 아래)
    chest_x = (ls_x + rs_x) / 2
    chest_y = (ls_y + rs_y) / 2 + 0.08
    shldr_w = abs(ls_x - rs_x)

    def near_chest(wx, wy):
        return (abs(wx - chest_x) < CHEST_THR + shldr_w * 0.15 and
                abs(wy - chest_y) < CHEST_THR)

    l_near = near_chest(lw_x, lw_y)
    r_near = near_chest(rw_x, rw_y)

    # ── 1순위: Landing ────────────────────────────────────────────
    if l_near and r_near:
        return {'label': 'Landing', 'color': COLOR_LANDING}

    # ── 2순위: Throttle(+) — 양쪽 손목 머리 위 ───────────────────
    if lw_y < nose_y and rw_y < nose_y:
        return {'label': 'Throttle(+)', 'color': COLOR_THROTTLE}

    # ── 3순위: Throttle(-) — 양쪽 손목 엉덩이 아래 ───────────────
    if lw_y > hip_y and rw_y > hip_y:
        return {'label': 'Throttle(-)', 'color': COLOR_THROTTLE}

    # ── 4순위: Yaw ────────────────────────────────────────────────
    if l_near and not r_near:
        return {'label': 'Yaw(-)', 'color': COLOR_YAW}
    if r_near and not l_near:
        return {'label': 'Yaw(+)', 'color': COLOR_YAW}

    # ── 5순위: Roll — 한쪽 팔만 어깨보다 위 ──────────────────────
    # 이미지 좌표: 위 = y 작음
    r_up = rw_y < rs_y
    l_up = lw_y < ls_y
    if r_up and not l_up:
        return {'label': 'Roll(+)', 'color': COLOR_ROLL}
    if l_up and not r_up:
        return {'label': 'Roll(-)', 'color': COLOR_ROLL}

    # ── 6순위: Takeoff/Hover — T자 ───────────────────────────────
    l_horiz = abs(lw_y - ls_y) < TPOSE_Y_THR
    r_horiz = abs(rw_y - rs_y) < TPOSE_Y_THR
    if l_horiz and r_horiz:
        return {'label': 'Takeoff/Hover', 'color': COLOR_TAKEOFF}

    return None

# ────────시각화 ───────────────────────────────────────────────
def draw_face_box(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_WHITE, 2)
        cv2.putText(frame, "User Detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_WHITE, 2, cv2.LINE_AA)


def draw_dwell_bar(frame, elapsed, total, fw, fh):
    ratio = min(elapsed / total, 1.0)
    bar_w = int(fw * 0.6)
    bx, by = (fw - bar_w) // 2, fh - 30
    cv2.rectangle(frame, (bx, by), (bx+bar_w, by+18), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx+int(bar_w*ratio), by+18), COLOR_WHITE, -1)
    cv2.putText(frame, f"Locking... {elapsed:.1f}s / {total:.0f}s",
                (bx, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)


def draw_wait_prompt(frame, fw, fh):
    text = "Stand in frame"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx, ty = (fw - tw) // 2, (fh + th) // 2
    cv2.rectangle(frame, (tx-10, ty-th-10), (tx+tw+10, ty+10), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), font, scale, COLOR_WHITE, thick, cv2.LINE_AA)


def draw_skeleton(frame, kps: dict):
    """어깨·손목·코 키포인트 + 연결선 시각화."""
    pts = {}
    for key, (x, y, c) in kps.items():
        if c >= KP_CONF:
            pts[key] = (int(x), int(y))
            cv2.circle(frame, pts[key], 5, COLOR_WHITE, -1)

    links = [
        ('l_shldr', 'r_shldr'),
        ('l_shldr', 'l_wrist'),
        ('r_shldr', 'r_wrist'),
        ('nose',    'l_shldr'),
        ('nose',    'r_shldr'),
        ('l_shldr', 'l_hip'),
        ('r_shldr', 'r_hip'),
    ]
    for a, b in links:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], COLOR_GRAY, 2)


def draw_gesture_label(frame, gesture: dict, fh: int):
    """왼쪽 하단에 제스처 레이블 표시."""
    label = gesture['label']
    color = gesture['color']
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    x, y = 20, fh - 20
    cv2.rectangle(frame, (x-6, y-th-10), (x+tw+6, y+8), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), font, scale, color, thick, cv2.LINE_AA)

# ────────메인 루프 ────────────────────────────────────────────
def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state      = STATE_FACE
    face_start = None
    print("시작. 종료: 'q'")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # ── 상태: 얼굴 감지 ──────────────────────────────────────
        if state == STATE_FACE:
            faces = detect_faces(frame)
            draw_face_box(frame, faces)

            if faces:
                if face_start is None:
                    face_start = time.time()
                elapsed = time.time() - face_start
                draw_dwell_bar(frame, elapsed, FACE_DWELL_SEC, fw, fh)
                if elapsed >= FACE_DWELL_SEC:
                    state      = STATE_WAIT_POSE
                    face_start = None
                    print("[전환] 얼굴 인식 완료 → 자세 인식 대기")
            else:
                face_start = None

        # ── 상태: 자세 인식 대기 ─────────────────────────────────
        elif state == STATE_WAIT_POSE:
            draw_wait_prompt(frame, fw, fh)
            kps = get_keypoints(frame)
            if kps is not None:
                draw_skeleton(frame, kps)
                state = STATE_GESTURE
                print("[전환] 자세 감지 완료 → 제스처 제어 시작")

        # ── 상태: 제스처 제어 ─────────────────────────────────────
        elif state == STATE_GESTURE:
            kps = get_keypoints(frame)
            if kps is not None:
                draw_skeleton(frame, kps)
                gesture = classify_gesture(kps, fw, fh)
                if gesture:
                    draw_gesture_label(frame, gesture, fh)
            else:
                cv2.putText(frame, "***조종자 자리 이탈***", (20, fh//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2, cv2.LINE_AA)

        # ── 공통 UI ───────────────────────────────────────────────
        cv2.putText(frame, f"State: {state}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow("Drone Body Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료.")


if __name__ == '__main__':
    run(camera_index=0)
