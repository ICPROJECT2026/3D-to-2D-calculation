import cv2
import numpy as np

# =========================
# Basic Config
# =========================
IMAGE_PATH = "/Users/jin/Desktop/table.jpg"
CAMERA_ID = 0
TABLE_WIDTH = 100.0
TABLE_HEIGHT = 60.0

WARP_VIEW_WIDTH = 800
WARP_VIEW_HEIGHT = 480

AUTO_DETECT_OBJECT = True
OBJECT_COLOR = "red"

# =========================
# Mode Selection
# =========================
print("Select input mode:")
print("1 = Camera")
print("2 = Image")
print("3 = Camera + manual corners")
print("4 = Image + manual corners only")
mode = input("Please enter 1, 2, 3 or 4: ").strip()

USE_CAMERA = False
AUTO_DETECT_TABLE = True

if mode == "1":
    USE_CAMERA = True
    AUTO_DETECT_TABLE = True

elif mode == "2":
    USE_CAMERA = False
    AUTO_DETECT_TABLE = True
    path_input = input("Please enter image path (press Enter to use default): ").strip()
    if path_input:
        IMAGE_PATH = path_input

elif mode == "3":
    USE_CAMERA = True
    AUTO_DETECT_TABLE = False

elif mode == "4":
    USE_CAMERA = False
    AUTO_DETECT_TABLE = False
    path_input = input("Please enter image path (press Enter to use default): ").strip()
    if path_input:
        IMAGE_PATH = path_input

else:
    print("Invalid input, defaulting to Image + manual corners only.")
    USE_CAMERA = False
    AUTO_DETECT_TABLE = False

# =========================
# Global State
# =========================
corner_points = []
object_points = []
H = None
H_inv = None
mouse_xy = None
current_frame = None
freeze_frame = None
use_frozen_frame = False

# real table size state
real_table_width = TABLE_WIDTH
real_table_height = TABLE_HEIGHT

# =========================
# Utility Functions
# =========================
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def compute_homography_from_corners(corners):
    global H, H_inv
    if len(corners) != 4:
        return False

    src_pts = order_points(corners)
    dst_pts = np.array([
        [0, 0],
        [real_table_width, 0],
        [real_table_width, real_table_height],
        [0, real_table_height]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return True

def image_to_table(pt_xy):
    global H
    if H is None:
        return None
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    x_real, y_real = mapped[0][0]
    return float(x_real), float(y_real)

def table_to_warp_pixel(table_xy):
    x, y = table_xy
    px = int(x / real_table_width * WARP_VIEW_WIDTH)
    py = int(y / real_table_height * WARP_VIEW_HEIGHT)
    return px, py

def warp_table_view(frame):
    global H
    if H is None or len(corner_points) != 4:
        return None

    dst_corners_pixels = np.array([
        [0, 0],
        [WARP_VIEW_WIDTH - 1, 0],
        [WARP_VIEW_WIDTH - 1, WARP_VIEW_HEIGHT - 1],
        [0, WARP_VIEW_HEIGHT - 1]
    ], dtype=np.float32)

    src_pts = order_points(corner_points)
    M = cv2.getPerspectiveTransform(src_pts, dst_corners_pixels)
    warped = cv2.warpPerspective(frame, M, (WARP_VIEW_WIDTH, WARP_VIEW_HEIGHT))
    return warped

def detect_table_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is not None:
        return order_points(best_quad)

    return None

def detect_red_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 80])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([160, 100, 80])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300 and area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is not None:
        M = cv2.moments(best_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), best_cnt, mask

    return None, None, mask

def draw_info_panel(img, lines, x=10, y=20, line_gap=24):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(img, line, (x, y + i * line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def print_all_object_real_points():
    if H is None:
        print("Homography not ready. Please finish table calibration first.")
        return

    if len(object_points) == 0:
        print("No clicked object points to convert.")
        return

    print("\n===== Converted Real Coordinates =====")
    print(f"Current table size: width={real_table_width:.2f}, height={real_table_height:.2f}")
    for idx, pt in enumerate(object_points):
        real_pt = image_to_table(pt)
        if real_pt is not None:
            rx, ry = real_pt
            print(f"P{idx+1}: pixel=({pt[0]}, {pt[1]}) -> real=({rx:.2f}, {ry:.2f})")
    print("=====================================\n")

def update_real_table_size():
    global real_table_width, real_table_height, H, H_inv

    try:
        w = float(input("Enter real table width: ").strip())
        h = float(input("Enter real table height: ").strip())

        if w <= 0 or h <= 0:
            print("Width and height must be positive.")
            return

        real_table_width = w
        real_table_height = h

        if len(corner_points) == 4:
            compute_homography_from_corners(corner_points)
            print(f"Updated real table size to width={real_table_width}, height={real_table_height}")
            print_all_object_real_points()
        else:
            print("Table size updated, but corners are not ready yet.")

    except ValueError:
        print("Invalid input. Please enter numeric values.")

def redraw_main_view(frame):
    display = frame.copy()

    for i, pt in enumerate(corner_points):
        pt_int = tuple(np.array(pt, dtype=int))
        cv2.circle(display, pt_int, 7, (0, 255, 0), -1)
        cv2.putText(display, f"C{i+1}", (pt_int[0] + 10, pt_int[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if len(corner_points) == 4:
        pts = np.array(corner_points, dtype=np.int32)
        cv2.polylines(display, [pts], True, (255, 0, 0), 2)

    for idx, pt in enumerate(object_points):
        x, y = pt
        cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display, f"P{idx+1}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        real_pt = image_to_table((x, y))
        if real_pt is not None:
            rx, ry = real_pt
            cv2.putText(display, f"({rx:.1f}, {ry:.1f})", (x + 10, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if mouse_xy is not None and H is not None:
        mx, my = mouse_xy
        real_pt = image_to_table((mx, my))
        if real_pt is not None:
            rx, ry = real_pt
            cv2.circle(display, (mx, my), 4, (0, 255, 255), -1)
            cv2.putText(display, f"Mouse: px=({mx},{my}) -> table=({rx:.1f},{ry:.1f})",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return display

def build_warp_view(frame, auto_object_result=None):
    warped = warp_table_view(frame)
    if warped is None:
        return None

    cv2.rectangle(warped, (0, 0), (WARP_VIEW_WIDTH - 1, WARP_VIEW_HEIGHT - 1), (255, 255, 255), 2)

    for idx, pt in enumerate(object_points):
        real_pt = image_to_table(pt)
        if real_pt is not None:
            wx, wy = table_to_warp_pixel(real_pt)
            cv2.circle(warped, (wx, wy), 6, (0, 0, 255), -1)
            cv2.putText(warped, f"P{idx+1}", (wx + 10, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(warped, f"({real_pt[0]:.1f}, {real_pt[1]:.1f})", (wx + 10, wy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if auto_object_result is not None:
        obj_center = auto_object_result
        real_pt = image_to_table(obj_center)
        if real_pt is not None:
            wx, wy = table_to_warp_pixel(real_pt)
            cv2.circle(warped, (wx, wy), 8, (0, 255, 255), -1)
            cv2.putText(warped, "AUTO_OBJ", (wx + 10, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(warped, f"({real_pt[0]:.1f}, {real_pt[1]:.1f})", (wx + 10, wy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if mouse_xy is not None and H is not None:
        real_pt = image_to_table(mouse_xy)
        if real_pt is not None:
            wx, wy = table_to_warp_pixel(real_pt)
            cv2.circle(warped, (wx, wy), 5, (255, 255, 0), -1)
            cv2.putText(warped, f"Mouse ({real_pt[0]:.1f}, {real_pt[1]:.1f})",
                        (10, WARP_VIEW_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return warped

# =========================
# Mouse Callback
# =========================
def mouse_callback(event, x, y, flags, param):
    global mouse_xy, corner_points, object_points

    mouse_xy = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            corner_points.append([x, y])
            print(f"Manual Corner {len(corner_points)}: ({x}, {y})")

            if len(corner_points) == 4:
                compute_homography_from_corners(corner_points)
                print("Homography computed from manual corners.")
        else:
            object_points.append([x, y])
            real_pt = image_to_table((x, y))
            if real_pt is not None:
                print(f"Clicked pixel: ({x}, {y}) -> Table coordinate: ({real_pt[0]:.2f}, {real_pt[1]:.2f})")

# =========================
# Main
# =========================
cv2.namedWindow("Main View")
cv2.setMouseCallback("Main View", mouse_callback)

if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
else:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {IMAGE_PATH}")

print("Controls:")
print("  q : quit")
print("  f : freeze/unfreeze current camera frame")
print("  a : auto-detect table corners")
print("  c : clear all corners and points")
print("  r : clear clicked object points only")
print("  m : manual corner reselection")
print("  t : input real table width/height and convert clicked points")
print("  p : print all clicked points real coordinates")
print("  s : save current main/warp/mask images")
print("")

while True:
    if USE_CAMERA:
        if use_frozen_frame and freeze_frame is not None:
            frame = freeze_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            current_frame = frame.copy()
    else:
        frame = img.copy()
        current_frame = frame.copy()

    auto_object_center = None
    object_contour = None
    object_mask = None

    if AUTO_DETECT_TABLE and len(corner_points) != 4:
        detected = detect_table_corners(frame)
        if detected is not None:
            corner_points = detected.tolist()
            compute_homography_from_corners(corner_points)

    if AUTO_DETECT_OBJECT:
        result, contour, mask = detect_red_object(frame)
        object_mask = mask
        if result is not None:
            auto_object_center = result
            object_contour = contour

    main_display = redraw_main_view(frame)

    if auto_object_center is not None:
        cx, cy = auto_object_center
        cv2.drawContours(main_display, [object_contour], -1, (0, 255, 255), 2)
        cv2.circle(main_display, (cx, cy), 8, (0, 255, 255), -1)
        real_pt = image_to_table((cx, cy))
        if real_pt is not None:
            cv2.putText(main_display, f"AUTO_OBJ ({real_pt[0]:.1f}, {real_pt[1]:.1f})",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    status_lines = [
        f"Mode: {'Camera' if USE_CAMERA else 'Image'}   AutoTable: {AUTO_DETECT_TABLE}   Frozen: {use_frozen_frame}",
        f"Corners: {len(corner_points)}/4   Clicked points: {len(object_points)}",
        f"Real Table Size: {real_table_width:.1f} x {real_table_height:.1f}",
        "Keys: q quit | f freeze | a auto table | c reset all | r clear points | m manual corners | t set table size | p print points | s save"
    ]
    draw_info_panel(main_display, status_lines)

    cv2.imshow("Main View", main_display)

    warp_display = build_warp_view(frame, auto_object_center)
    if warp_display is not None:
        cv2.imshow("Warped Top-Down View", warp_display)

    if object_mask is not None:
        cv2.imshow("Object Mask", object_mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('f'):
        if USE_CAMERA:
            use_frozen_frame = not use_frozen_frame
            if use_frozen_frame and current_frame is not None:
                freeze_frame = current_frame.copy()
            print(f"Frozen frame mode: {use_frozen_frame}")

    elif key == ord('a'):
        detected = detect_table_corners(frame)
        if detected is not None:
            corner_points = detected.tolist()
            compute_homography_from_corners(corner_points)
            print("Auto-detected table corners.")
        else:
            print("Auto-detect table failed.")

    elif key == ord('c'):
        corner_points = []
        object_points = []
        H = None
        H_inv = None
        print("Cleared all corners and points.")

    elif key == ord('r'):
        object_points = []
        print("Cleared clicked object points.")

    elif key == ord('m'):
        corner_points = []
        H = None
        H_inv = None
        print("Manual corner reselection enabled. Please click 4 corners.")

    elif key == ord('t'):
        update_real_table_size()

    elif key == ord('p'):
        print_all_object_real_points()

    elif key == ord('s'):
        cv2.imwrite("main_view.png", main_display)
        print("Saved: main_view.png")
        if warp_display is not None:
            cv2.imwrite("warped_view.png", warp_display)
            print("Saved: warped_view.png")
        if object_mask is not None:
            cv2.imwrite("object_mask.png", object_mask)
            print("Saved: object_mask.png")

if USE_CAMERA:
    cap.release()

cv2.destroyAllWindows()
