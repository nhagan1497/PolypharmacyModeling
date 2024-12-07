from PIL import Image
from ultralytics import YOLO

def get_pill_properties(image: Image, yolo_model: YOLO):
    def calculate_overlap(box1, box2):
        box1 = [float(num) for num in box1]
        box2 = [float(num) for num in box2]

        # Extract coordinates of the boxes
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Compute intersection coordinates
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        # Calculate intersection area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Calculate areas of both boxes
        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (x4 - x3) * (y4 - y3)

        # Calculate union area
        union_area = area_box1 + area_box2 - inter_area

        # Handle edge case: if union area is 0
        if union_area == 0:
            return 0

        # Calculate overlap percentage
        overlap_percentage = (inter_area / union_area)
        return round(float(overlap_percentage), 4)

    def is_box_inside(inner_box, outer_box):
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box

        # Check if all corners of the inner box are within the outer box
        return (
                inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and  # Top-left corner inside
                inner_x2 <= outer_x2 and inner_y2 <= outer_y2  # Bottom-right corner inside
        )

    results = yolo_model(image.resize((640, 640)), conf=0.1)
    result = results[0]

    pills = {box: [] for box in result.boxes if result.names[int(box.cls)] == 'Pill'}
    overlapped_pills = set()
    for pill in pills:
        for inner_pill in pills:
            if pill == inner_pill:
                continue
            if is_box_inside(pill.xyxy[0], inner_pill.xyxy[0]):
                overlapped_pills.add(pill)
    for pill in overlapped_pills:
        del pills[pill]

    overlapped_pills = set()
    for pill in pills:
        for inner_pill in pills:
            if pill == inner_pill:
                continue
            if calculate_overlap(pill.xyxy[0], inner_pill.xyxy[0]) > 0.75:
                print(calculate_overlap(pill.xyxy[0], inner_pill.xyxy[0]))
                overlapped_pills.add(pill)
    for pill in overlapped_pills:
        del pills[pill]

    for box in result.boxes:
        if result.names[int(box.cls)] == 'Pill':
            continue
        for pill in pills:
            if calculate_overlap(box.xyxy[0], pill.xyxy[0]) > 0.25:
                pills[pill].append(box)

    pill_property_list = []

    for pill, properties in pills.items():
        pill_property_list.append([result.names[int(prop.cls)] for prop in properties])

    return pill_property_list
