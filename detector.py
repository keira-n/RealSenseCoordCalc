from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    def detect(self, color_image, confidence=0.5,
               target_classes=None):

        results = self.model(
            color_image,
            conf=confidence,
            verbose=False
        )

        height, width = color_image.shape[:2]
        objects = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence_score = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Class filtering
                if (target_classes is not None
                        and class_name not in target_classes):
                    continue

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Validate center pixel
                if not (0 <= center_x < width
                        and 0 <= center_y < height):
                    continue

                objects.append({
                    "name": class_name,
                    "class_id": class_id,
                    "confidence": confidence_score,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "center": (
                        center_x,
                        center_y
                    )
                })

        # Highest-confidence detection first
        objects.sort(
            key=lambda obj: obj["confidence"],
            reverse=True
        )

        return objects