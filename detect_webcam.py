import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecção de EPIs (Helmet / Glasses) em tempo real com YOLO"
    )
    parser.add_argument("--weights", type=str, required=True, help="Caminho do modelo (.pt)")
    parser.add_argument("--device", type=str, default="0", help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução de inferência")
    return parser.parse_args()


def draw_hud(frame, helmet_ok, glasses_ok):
    """Painel no canto inferior direito mostrando True/False."""
    h, w = frame.shape[:2]

    pw, ph = 260, 70
    margin = 10

    x2, y2 = w - margin, h - margin
    x1, y1 = x2 - pw, y2 - ph

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    th = 2

    cv2.putText(frame, f"Helmet: {helmet_ok}", (x1 + 10, y1 + 25),
                font, fs, (0,255,0) if helmet_ok else (0,0,255), th, cv2.LINE_AA)

    cv2.putText(frame, f"Glasses: {glasses_ok}", (x1 + 10, y1 + 50),
                font, fs, (0,255,0) if glasses_ok else (0,0,255), th, cv2.LINE_AA)


def main():
    args = parse_args()
    model = YOLO(args.weights)

    print("[INFO] Classes carregadas:")
    for cid, cname in model.names.items():
        print(f"  {cid}: {cname}")

    # ------------------------------------------------------------
    # MAPEAMENTO ESPECÍFICO DAS CLASSES
    # ------------------------------------------------------------
    CLASS_PERSON = None
    CLASS_HELMET = None
    CLASS_GLASSES = None

    for cid, cname in model.names.items():
        name = cname.lower()

        if name == "person":
            CLASS_PERSON = cid
        if name == "helmet":
            CLASS_HELMET = cid
        if name == "glasses":
            CLASS_GLASSES = cid

    print("\n[INFO] Classes importantes:")
    print("  person:", CLASS_PERSON)
    print("  helmet:", CLASS_HELMET)
    print("  glasses:", CLASS_GLASSES)
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Webcam não encontrada.")
        return

    print("[INFO] Pressione ESC para sair...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        det = results[0]
        boxes = det.boxes

        helmet = False
        glasses = False
        person_found = False

        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls]

            # Filtrar — aceitar SOMENTE:
            # person, helmet, glasses
            if cls not in (CLASS_PERSON, CLASS_HELMET, CLASS_GLASSES):
                continue

            # Atualizar status
            if cls == CLASS_HELMET:
                helmet = True
            if cls == CLASS_GLASSES:
                glasses = True
            if cls == CLASS_PERSON:
                person_found = True

            # Cor:
            color = (0, 255, 0) if cls in (CLASS_HELMET, CLASS_GLASSES) else (0, 165, 255)

            # Caixas
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # HUD só aparece se houver pessoa detectada
        if person_found:
            draw_hud(frame, helmet, glasses)

        cv2.imshow("Real-time PPE Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
