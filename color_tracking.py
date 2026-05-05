import cv2
import numpy as np
import argparse
from tqdm import tqdm

# ---------------- Global state ----------------
rectangles = []
drawing = False
ix, iy = -1, -1
img = None
img_copy = None
selected_points = []

time_dragging = False
time_x1 = None
time_x2 = None


# ---------------- Window helpers ----------------
def create_window(name, width=None, height=None):
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        flags |= cv2.WINDOW_KEEPRATIO
    if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
        flags |= cv2.WINDOW_GUI_EXPANDED

    cv2.namedWindow(name, flags)

    if width is not None and height is not None:
        cv2.resizeWindow(name, width, height)


# ---------------- Key helpers ----------------
def is_left_key(key):
    return key in (81, 2424832)

def is_right_key(key):
    return key in (83, 2555904)

def is_up_key(key):
    return key in (82, 2490368)

def is_down_key(key):
    return key in (84, 2621440)


# ---------------- Display helpers ----------------
def fit_image_to_window(image, window_name, fallback_w=1400, fallback_h=900, allow_upscale=False):
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        if win_w <= 0 or win_h <= 0:
            raise ValueError
    except Exception:
        h, w = image.shape[:2]
        win_w, win_h = min(w, fallback_w), min(h, fallback_h)

    h, w = image.shape[:2]
    max_scale = 10.0 if allow_upscale else 1.0
    scale = min(win_w / w, win_h / h, max_scale)

    display_w = max(1, int(w * scale))
    display_h = max(1, int(h * scale))

    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (display_w, display_h), interpolation=interp)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    offset_x = (win_w - display_w) // 2
    offset_y = (win_h - display_h) // 2
    canvas[offset_y:offset_y + display_h, offset_x:offset_x + display_w] = resized

    display_info = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "display_w": display_w,
        "display_h": display_h,
        "canvas_w": win_w,
        "canvas_h": win_h
    }
    return canvas, display_info


def make_display_canvas(frame, max_width=1400, max_height=900):
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    offset_x = (max_width - new_w) // 2
    offset_y = (max_height - new_h) // 2

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    display_info = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "display_w": new_w,
        "display_h": new_h,
    }

    return canvas, display_info


def fit_frame_to_window(frame, window_name):
    return fit_image_to_window(frame, window_name, fallback_w=1400, fallback_h=900, allow_upscale=False)


def make_zoom_view(frame, zoom, center_x=None, center_y=None, window_name="Zoom View"):
    h, w = frame.shape[:2]

    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        if win_w <= 0 or win_h <= 0:
            raise ValueError
    except Exception:
        win_w, win_h = min(w, 1400), min(h, 900)

    crop_w = max(50, int(w / zoom))
    crop_h = max(50, int(h / zoom))

    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2

    center_x = max(crop_w // 2, min(center_x, w - crop_w // 2))
    center_y = max(crop_h // 2, min(center_y, h - crop_h // 2))

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    cropped = frame[y1:y2, x1:x2]
    canvas, base_info = fit_image_to_window(cropped, window_name, fallback_w=1400, fallback_h=900, allow_upscale=True)

    display_info = {
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        **base_info
    }

    return canvas, display_info


def display_to_original(x, y, display_info):
    offset_x = display_info["offset_x"]
    offset_y = display_info["offset_y"]
    display_w = display_info["display_w"]
    display_h = display_info["display_h"]

    if not (offset_x <= x < offset_x + display_w and offset_y <= y < offset_y + display_h):
        return None

    if "crop_x1" in display_info:
        local_x = (x - offset_x) / display_info["scale"]
        local_y = (y - offset_y) / display_info["scale"]
        original_x = int(display_info["crop_x1"] + local_x)
        original_y = int(display_info["crop_y1"] + local_y)
    else:
        original_x = int((x - offset_x) / display_info["scale"])
        original_y = int((y - offset_y) / display_info["scale"])

    return original_x, original_y


def original_to_display(x, y, display_info):
    if "crop_x1" in display_info:
        x = x - display_info["crop_x1"]
        y = y - display_info["crop_y1"]

    dx = int(x * display_info["scale"] + display_info["offset_x"])
    dy = int(y * display_info["scale"] + display_info["offset_y"])
    return dx, dy


def display_to_ui(x, y, display_info):
    offset_x = display_info["offset_x"]
    offset_y = display_info["offset_y"]
    display_w = display_info["display_w"]
    display_h = display_info["display_h"]

    if not (offset_x <= x < offset_x + display_w and offset_y <= y < offset_y + display_h):
        return None

    ui_x = int((x - offset_x) / display_info["scale"])
    ui_y = int((y - offset_y) / display_info["scale"])
    return ui_x, ui_y


def ui_to_display(x, y, display_info):
    dx = int(x * display_info["scale"] + display_info["offset_x"])
    dy = int(y * display_info["scale"] + display_info["offset_y"])
    return dx, dy


def clamp_to_display_area(x, y, display_info):
    offset_x = display_info["offset_x"]
    offset_y = display_info["offset_y"]
    display_w = display_info["display_w"]
    display_h = display_info["display_h"]

    x = max(offset_x, min(x, offset_x + display_w - 1))
    y = max(offset_y, min(y, offset_y + display_h - 1))
    return x, y


def hsv_to_bgr_color(hsv_color):
    hsv_pixel = np.uint8([[hsv_color]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(v) for v in bgr_pixel)


# ---------------- Time range selection in seconds ----------------
def time_range_mouse_callback(event, x, y, flags, param):
    global time_dragging, time_x1, time_x2

    get_display_info = param["display_info_getter"]
    timeline_left = param["timeline_left"]
    timeline_right = param["timeline_right"]
    timeline_y1 = param["timeline_y1"]
    timeline_y2 = param["timeline_y2"]

    display_info = get_display_info()
    if display_info is None:
        return

    ui_point = display_to_ui(x, y, display_info)
    if ui_point is None:
        return

    ux, uy = ui_point
    inside_timeline = timeline_left <= ux <= timeline_right and timeline_y1 <= uy <= timeline_y2

    if event == cv2.EVENT_LBUTTONDOWN and inside_timeline:
        time_dragging = True
        time_x1 = ux
        time_x2 = ux

    elif event == cv2.EVENT_MOUSEMOVE and time_dragging:
        time_x2 = max(timeline_left, min(ux, timeline_right))

    elif event == cv2.EVENT_LBUTTONUP and time_dragging:
        time_dragging = False
        time_x2 = max(timeline_left, min(ux, timeline_right))


def seconds_to_frame(seconds, fps, total_frames):
    frame = int(seconds * fps)
    return max(0, min(frame, total_frames - 1))


def select_timeframe(video_path):
    global time_dragging, time_x1, time_x2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        print("Invalid FPS.")
        return None, None

    total_seconds = total_frames / fps

    ui_w = 1400
    ui_h = 360
    timeline_left = 90
    timeline_right = ui_w - 90
    timeline_y1 = 180
    timeline_y2 = 220
    timeline_w = timeline_right - timeline_left

    time_dragging = False
    time_x1 = None
    time_x2 = None
    current_display_info = None

    def get_display_info():
        return current_display_info

    cv2.destroyAllWindows()
    create_window("Step 0 - Select Time Range", width=1200, height=500)
    cv2.setMouseCallback(
        "Step 0 - Select Time Range",
        time_range_mouse_callback,
        {
            "display_info_getter": get_display_info,
            "timeline_left": timeline_left,
            "timeline_right": timeline_right,
            "timeline_y1": timeline_y1,
            "timeline_y2": timeline_y2
        }
    )

    print("Step 0 - Time range selection:")
    print("Drag with mouse on the timeline to choose start and end time.")
    print("Press ENTER to confirm, 'r' to reset, ESC to cancel.")

    while True:
        ui = np.full((ui_h, ui_w, 3), 28, dtype=np.uint8)

        cv2.putText(
            ui,
            "Step 0: Drag on the timeline to choose a time range in seconds",
            (60, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (240, 240, 240),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            ui,
            f"Video length: {total_seconds:.2f} s",
            (60, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(ui, (timeline_left, timeline_y1), (timeline_right, timeline_y2), (70, 70, 70), -1)
        cv2.rectangle(ui, (timeline_left, timeline_y1), (timeline_right, timeline_y2), (140, 140, 140), 2)

        if time_x1 is not None and time_x2 is not None:
            x1 = max(timeline_left, min(time_x1, time_x2))
            x2 = min(timeline_right, max(time_x1, time_x2))
            cv2.rectangle(ui, (x1, timeline_y1), (x2, timeline_y2), (0, 200, 255), -1)

            start_ratio = (x1 - timeline_left) / timeline_w
            end_ratio = (x2 - timeline_left) / timeline_w

            start_sec = start_ratio * total_seconds
            end_sec = end_ratio * total_seconds

            cv2.putText(
                ui,
                f"Selected: {start_sec:.2f} s  ->  {end_sec:.2f} s",
                (60, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            ui,
            "0.00 s",
            (timeline_left - 20, timeline_y2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (210, 210, 210),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            ui,
            f"{total_seconds:.2f} s",
            (timeline_right - 95, timeline_y2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (210, 210, 210),
            2,
            cv2.LINE_AA
        )

        for frac in np.linspace(0, 1, 11):
            x = int(timeline_left + frac * timeline_w)
            cv2.line(ui, (x, timeline_y2 + 4), (x, timeline_y2 + 16), (160, 160, 160), 1)
            sec = frac * total_seconds
            cv2.putText(
                ui,
                f"{sec:.0f}",
                (x - 10, timeline_y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
                cv2.LINE_AA
            )

        canvas, current_display_info = fit_image_to_window(
            ui,
            "Step 0 - Select Time Range",
            fallback_w=1200,
            fallback_h=500,
            allow_upscale=True
        )

        cv2.imshow("Step 0 - Select Time Range", canvas)
        key = cv2.waitKeyEx(20)

        if key == 13:
            if time_x1 is None or time_x2 is None:
                print("Please choose a time range first.")
                continue

            x1 = max(timeline_left, min(time_x1, time_x2))
            x2 = min(timeline_right, max(time_x1, time_x2))

            start_ratio = (x1 - timeline_left) / timeline_w
            end_ratio = (x2 - timeline_left) / timeline_w

            start_sec = start_ratio * total_seconds
            end_sec = end_ratio * total_seconds

            if abs(end_sec - start_sec) < 1.0 / fps:
                end_sec = min(total_seconds, start_sec + 1.0 / fps)

            start_frame = seconds_to_frame(start_sec, fps, total_frames)
            end_frame = seconds_to_frame(end_sec, fps, total_frames)

            if start_frame == end_frame:
                end_frame = min(total_frames - 1, start_frame + 1)

            cv2.destroyAllWindows()
            return start_frame, end_frame

        elif key == ord("r"):
            time_x1 = None
            time_x2 = None

        elif key == 27:
            cv2.destroyAllWindows()
            return None, None


# ---------------- Mouse callbacks for later steps ----------------
def click_color_point(event, x, y, flags, param):
    global selected_points

    display_info = param["display_info_getter"]()
    if display_info is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        original_point = display_to_original(x, y, display_info)
        if original_point is None:
            print("Clicked outside the video frame.")
            return

        selected_points.append(original_point)


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, img, img_copy

    display_info = param["display_info"]

    if event == cv2.EVENT_LBUTTONDOWN:
        original_point = display_to_original(x, y, display_info)
        if original_point is None:
            print("Start point outside video frame.")
            return
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x_clamped, y_clamped = clamp_to_display_area(x, y, display_info)
        temp = img_copy.copy()
        cv2.rectangle(temp, (ix, iy), (x_clamped, y_clamped), (0, 255, 0), 2)
        img[:] = temp

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x_clamped, y_clamped = clamp_to_display_area(x, y, display_info)
        cv2.rectangle(img, (ix, iy), (x_clamped, y_clamped), (0, 255, 0), 2)

        p1 = display_to_original(ix, iy, display_info)
        p2 = display_to_original(x_clamped, y_clamped, display_info)

        if p1 is None or p2 is None:
            return

        x1, y1 = p1
        x2, y2 = p2
        rectangles.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        img_copy[:] = img


# ---------------- Frame selection ----------------
def select_frame_from_range(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return None

    idx = start_frame
    selected_frame = None
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.destroyAllWindows()
    create_window("Step 1 - Select Frame", width=1200, height=800)

    print("Step 1 - Frame selection:")
    print("Use LEFT and RIGHT arrow keys to move through the selected time range.")
    print("Press ENTER to choose this frame.")
    print("Press ESC to cancel.")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        canvas, _ = fit_frame_to_window(frame, "Step 1 - Select Frame")

        current_sec = idx / fps
        start_sec = start_frame / fps
        end_sec = end_frame / fps

        cv2.putText(
            canvas,
            f"Time {current_sec:.2f}s  |  Range: {start_sec:.2f}s - {end_sec:.2f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Step 1 - Select Frame", canvas)
        key = cv2.waitKeyEx(0)

        if is_right_key(key):
            idx = min(idx + 1, end_frame)
        elif is_left_key(key):
            idx = max(idx - 1, start_frame)
        elif key == 13:
            selected_frame = frame.copy()
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return selected_frame


# ---------------- HSV point selection ----------------
def select_hsv_points_from_frame(selected_frame):
    global img, img_copy, selected_points
    selected_points = []

    zoom = 1.0
    zoom_center_x = selected_frame.shape[1] // 2
    zoom_center_y = selected_frame.shape[0] // 2
    current_display_info = None

    def get_display_info():
        return current_display_info

    cv2.destroyAllWindows()
    create_window("Step 2 - Select Colors", width=1200, height=800)
    cv2.setMouseCallback(
        "Step 2 - Select Colors",
        click_color_point,
        {"display_info_getter": get_display_info}
    )

    print("Step 2 - HSV color selection:")
    print("Left click: select color point")
    print("'+/=' zoom in, '-/_' zoom out")
    print("Arrow keys: move/pan inside the zoomed view")
    print("'r' reset points, ENTER finish, ESC cancel")

    while True:
        canvas, display_info = make_zoom_view(
            selected_frame,
            zoom,
            zoom_center_x,
            zoom_center_y,
            "Step 2 - Select Colors"
        )
        current_display_info = display_info

        img = canvas.copy()
        img_copy = canvas.copy()

        for i, (px, py) in enumerate(selected_points, start=1):
            if (
                display_info["crop_x1"] <= px < display_info["crop_x2"]
                and display_info["crop_y1"] <= py < display_info["crop_y2"]
            ):
                dx, dy = original_to_display(px, py, display_info)
                cv2.circle(img, (dx, dy), 6, (0, 255, 255), -1)
                cv2.putText(
                    img,
                    str(i),
                    (dx + 5, dy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        cv2.putText(
            img,
            f"Zoom: {zoom:.1f}x",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Step 2 - Select Colors", img)
        key = cv2.waitKeyEx(30)

        crop_w = max(50, int(selected_frame.shape[1] / zoom))
        crop_h = max(50, int(selected_frame.shape[0] / zoom))
        pan_x = max(10, crop_w // 8)
        pan_y = max(10, crop_h // 8)

        if key == 13:
            break
        elif key == ord("r"):
            selected_points = []
        elif key in (ord("+"), ord("=")):
            zoom = min(20.0, zoom * 1.25)
        elif key in (ord("-"), ord("_")):
            zoom = max(1.0, zoom / 1.25)
        elif is_left_key(key):
            zoom_center_x = max(0, zoom_center_x - pan_x)
        elif is_right_key(key):
            zoom_center_x = min(selected_frame.shape[1] - 1, zoom_center_x + pan_x)
        elif is_up_key(key):
            zoom_center_y = max(0, zoom_center_y - pan_y)
        elif is_down_key(key):
            zoom_center_y = min(selected_frame.shape[0] - 1, zoom_center_y + pan_y)
        elif key == 27:
            selected_points = []
            break

        if selected_points:
            zoom_center_x, zoom_center_y = selected_points[-1]

    cv2.destroyAllWindows()

    if not selected_points:
        print("No color points selected.")
        return []

    hsv_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2HSV)
    hsv_colors = []

    for x, y in selected_points:
        hsv_value = hsv_frame[y, x]
        hsv_colors.append(hsv_value)
        print(f"Selected point ({x}, {y}) HSV: {hsv_value}")

    return hsv_colors


# ---------------- ROI selection ----------------
def multi_roi_selector(frame_for_roi):
    global img, img_copy, rectangles
    rectangles = []

    canvas, display_info = make_display_canvas(frame_for_roi, max_width=1400, max_height=900)

    img = canvas.copy()
    img_copy = canvas.copy()

    cv2.destroyAllWindows()
    create_window("Step 3 - ROI Select", width=1200, height=800)
    cv2.imshow("Step 3 - ROI Select", img)

    cv2.setMouseCallback(
        "Step 3 - ROI Select",
        draw_rectangle,
        {"display_info": display_info}
    )

    print("Step 3 - ROI selection:")
    print("Draw one or more ROIs.")
    print("Press ENTER when done, 'r' to reset, or ESC to cancel.")

    while True:
        cv2.imshow("Step 3 - ROI Select", img)
        key = cv2.waitKeyEx(1)

        if key == 13:
            break
        elif key == ord("r"):
            rectangles = []
            img = canvas.copy()
            img_copy = canvas.copy()
        elif key == 27:
            rectangles = []
            break

    cv2.destroyAllWindows()
    return rectangles


# ---------------- Processing ----------------
class LaserTracker:
    def __init__(self, input_video, output_video, hsv_colors, slow_factor=2, sensitivity=50, rois=None):
        self.cap = cv2.VideoCapture(input_video)

        if not self.cap.isOpened():
            raise ValueError("Could not open input video.")

        self.hsv_colors = hsv_colors
        self.overlay_colors = [hsv_to_bgr_color(color) for color in hsv_colors]
        self.sensitivity = max(0, min(100, sensitivity))
        self.rois = rois
        self.slow_factor = slow_factor

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        output_fps = max(1, int(fps // slow_factor))

        self.out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height)
        )

    def create_mask(self, roi_frame, hsv_color):
        h, s, v = [int(x) for x in hsv_color]

        hue_margin = int(self.sensitivity * 0.4)
        sat_margin = int(self.sensitivity * 1.5)
        val_margin = int(self.sensitivity * 1.5)

        lower_sat = max(0, s - sat_margin)
        lower_val = max(0, v - val_margin)

        lower_hue = h - hue_margin
        upper_hue = h + hue_margin

        if 0 <= lower_hue and upper_hue <= 180:
            lower = np.array([lower_hue, lower_sat, lower_val])
            upper = np.array([upper_hue, 255, 255])
            return cv2.inRange(roi_frame, lower, upper)

        if lower_hue < 0:
            mask1 = cv2.inRange(
                roi_frame,
                np.array([0, lower_sat, lower_val]),
                np.array([upper_hue, 255, 255])
            )
            mask2 = cv2.inRange(
                roi_frame,
                np.array([180 + lower_hue, lower_sat, lower_val]),
                np.array([180, 255, 255])
            )
            return cv2.bitwise_or(mask1, mask2)

        mask1 = cv2.inRange(
            roi_frame,
            np.array([lower_hue, lower_sat, lower_val]),
            np.array([180, 255, 255])
        )
        mask2 = cv2.inRange(
            roi_frame,
            np.array([0, lower_sat, lower_val]),
            np.array([upper_hue - 180, 255, 255])
        )
        return cv2.bitwise_or(mask1, mask2)

    def process(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for roi in self.rois:
                x1, y1, x2, y2 = roi

                if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                    continue

                roi_frame = hsv[y1:y2, x1:x2]

                for i, hsv_color in enumerate(self.hsv_colors):
                    mask = self.create_mask(roi_frame, hsv_color)

                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x) + x1, int(y) + y1)

                        overlay = frame.copy()
                        cv2.circle(
                            overlay,
                            center,
                            8,
                            self.overlay_colors[i],
                            -1
                        )
                        cv2.addWeighted(overlay, 0.8, frame, 0.55, 0, frame)

            for _ in range(self.slow_factor):
                self.out.write(frame)

            progress_bar.update(1)

        progress_bar.close()
        self.cap.release()
        self.out.release()


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General HSV-based laser enhancement from selected video frame points."
    )

    parser.add_argument("input_video")
    parser.add_argument("output_video")
    parser.add_argument("--slow_factor", type=int, default=2)
    parser.add_argument("--sensitivity", type=int, default=50)

    args = parser.parse_args()

    start_frame, end_frame = select_timeframe(args.input_video)
    if start_frame is None or end_frame is None:
        raise ValueError("No time range selected.")

    selected_frame = select_frame_from_range(args.input_video, start_frame, end_frame)
    if selected_frame is None:
        raise ValueError("No frame selected.")

    hsv_colors = select_hsv_points_from_frame(selected_frame)
    if not hsv_colors:
        raise ValueError("No HSV colors selected.")

    rois = multi_roi_selector(selected_frame)
    if not rois:
        raise ValueError("No ROIs selected.")

    tracker = LaserTracker(
        args.input_video,
        args.output_video,
        hsv_colors=hsv_colors,
        slow_factor=args.slow_factor,
        sensitivity=args.sensitivity,
        rois=rois
    )

    tracker.process()
