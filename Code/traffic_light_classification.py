import cv2
import numpy as np
import torch

CIRCLE_IMAGE_PATH = "Models/traffic_light_templates/template_circle.jpg"
ARROW_IMAGE_PATH = "Models/traffic_light_templates/template_arrow.jpg"

def classify_light(image, box):
    details = {'type': 'traffic'}

    im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    xmin, ymin, xmax, ymax = map(int, box.tolist())
    
    w = (xmax - xmin)
    v = (ymax - ymin)

    if w*v < 240:
        details['qt'] = 0
        return details

    aspect_ratio = abs(v/w)

    if aspect_ratio < 1.2:
        # we are rendering square lights a 3 by 1 lights still
        details["qt"] = 1
        qt = 'square'
    elif aspect_ratio < 1.7:
        details["qt"] = 2
        qt = 'double'
    else:
        details["qt"] = 1
        qt = 'regular'

    border = w // 10

    cropped_im = im[(ymin + border):(ymax-border), (xmin + border):(xmax - border)]

    height, width = cropped_im.shape

    if qt == 'square':
        template_width = min(height, width)
        
    if qt == 'regular':
        green_start = height - width
        yellow_start = (green_start // 2)

        red_crop = cropped_im[:width]
        green_crop = cropped_im[green_start:]
        yellow_crop = cropped_im[yellow_start: yellow_start + width]

        crops = {"RED_": red_crop, "YELLOW_": yellow_crop, "GREEN_": green_crop}
        crops_list = [crops]

        template_width = width

    elif qt == 'double':
        hw = width//2
        hw_rounded_up = (width + 1)//2

        green_start = height - hw
        yellow_start = (green_start // 2)

        red_crop_l = cropped_im[:hw, :hw]
        green_crop_l = cropped_im[green_start:, :hw]
        yellow_crop_l = cropped_im[yellow_start: (yellow_start + hw), :hw]
        red_crop_r = cropped_im[:hw, hw_rounded_up:]
        green_crop_r = cropped_im[green_start:, hw_rounded_up:]
        yellow_crop_r = cropped_im[yellow_start: (yellow_start + hw), hw_rounded_up:]

        crop_left = {"RED_": red_crop_l, "YELLOW_": yellow_crop_l, "GREEN_": green_crop_l}
        crop_right = {"RED_": red_crop_r, "YELLOW_": yellow_crop_r, "GREEN_": green_crop_r}
        crops_list = [crop_left, crop_right]

        template_width = hw


    master_templates = {
            "circle_sm": create_circle_template(template_width, radius=0.25),
            "circle_md": create_circle_template(template_width, radius=0.28),
            "circle_lg": create_circle_template(template_width, radius=0.33),
            "circle_xl": create_circle_template(template_width, radius=0.4),
            "circle_img": template_from_img(template_width, path='template_circle.jpg'),
            "arrow_left": template_from_img(template_width, path='template_arrow.jpg'),
            "arrow_up": cv2.rotate(template_from_img(template_width, path='template_arrow.jpg'), cv2.ROTATE_90_CLOCKWISE),
            "arrow_right": cv2.flip(template_from_img(template_width, path='template_arrow.jpg'), 1),
            "arrow_down": cv2.rotate(template_from_img(template_width, path='template_arrow.jpg'), cv2.ROTATE_90_COUNTERCLOCKWISE)
        }
    
    if qt == "square":
        light = {}
        label, score = get_best_match(cropped_im, master_templates)
        if score > 0.5:
            results = (label, score)

            hsv_im = cv2.cvtColor(image[(ymin + border):(ymax-border), (xmin + border):(xmax - border)], cv2.COLOR_BGR2HSV)
            mask = (master_templates[label] > 0).astype(np.uint8)

            mask = cv2.resize(mask, (hsv_im.shape[1], hsv_im.shape[0]), interpolation=cv2.INTER_NEAREST)

            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)

            avg_hsv = cv2.mean(hsv_im, mask=mask)[:3]

            h, s, v = avg_hsv

            # Red: 0-10 or 160-180 | Yellow: 15-35 | Green: 40-90
            if s < 70 or v < 100: 
                active_color = "unknown/off"
            elif (h < 8) or (h > 165): # Tightened Red (Standard is 0-10, 160-180)
                active_color = "RED_"
            elif 20 < h < 35:          # Tightened Yellow (Standard is 15-45)
                active_color = "YELLOW_"
            elif 55 < h < 90:          # Tightened Green (Standard is 40-100)
                active_color = "GREEN_"
            else:
                active_color = "unknown/off"

            light["color"] = active_color
            light["shape"] = label

            details["light_0"] = light
            print(details)
            return details
        else:
            details['qt'] = 0
            return details

            
    results = {}
    for i, crops in enumerate(crops_list):
        light = {}
        for color, img in crops.items():
            label, score = get_best_match(img, master_templates)
            # A score below 0.5 usually means the light is 'off'
            if score > 0.5:
                results[color] = (label, score)

            
        # Find which color has the highest confidence score
        if results:
            sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
            best_color, (best_shape, best_val) = sorted_results[0]

            # Check if the gap to the 2nd and 3rd place is at least 0.15
            is_ambiguous = any((best_val - res[1][1]) < 0.15 for res in sorted_results[1:])

            if is_ambiguous:
                light.update({"color": "unknown", "shape": "unknown/off"})
            else:
                light.update({"color": best_color, "shape": best_shape})
        else:
            light["color"] = "unknown/off"
            light["shape"] = "unknown/off"

        details[f"light_{i}"] = light

    print(details)

    return details

        

def get_best_match(crop, templates, padding=4):
    """
    crops: a 2d crop of the light
    templates: a dict {'circle': mask1, 'arrow_up': mask2, ...}
    """
    best_score = -1
    best_label = "off"

    search_area = cv2.copyMakeBorder(
        crop, 
        padding, padding, padding, padding, 
        cv2.BORDER_REPLICATE
    )

    for label, template in templates.items():

        # Perform matching
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label, best_score


def create_circle_template(size=25, radius=0.33):
    # Create a black square
    template = np.zeros((size, size), dtype=np.uint8)
    # Draw a white filled circle in the center
    center = size // 2
    radius = int(size * radius) # Leave a small margin
    cv2.circle(template, (center, center), radius, 255, -1)

    k_size = int(size * 0.08) # responsive kernel size (around 5 for 60px)
    if k_size % 2 == 0: k_size += 1 # must be odd
    soft_template = cv2.GaussianBlur(template, (k_size, k_size), 0)
    
    return soft_template


# def create_arrow_template(size=25):
#     # 1. Start with a black canvas
#     template = np.zeros((size, size), dtype=np.uint8)
    
#     # 2. Define "short and fat" geometry as fractions of the 'size'
#     # We make the arrow occupy a lot of the width, but keep it short vertically
#     mid = size / 2
#     top_limit = size * 0.1     # Lowered (making it shorter)
#     bottom_limit = size * 0.6   # Raised (making it shorter)
    
#     width_outer = size * 0.3    # Wide wings (occupies 90% of width)
#     width_inner = size * 0.15    # Wide stem (making it "fat")
#     wing_y = size * 0.4         # Point where triangle meets stem

#     # 3. Calculate the 7 polygon points
#     pts = np.array([
#         [mid, top_limit],                    # Tip
#         [mid - width_outer, wing_y],         # Left Wing
#         [mid - width_inner, wing_y],         # Left Notch
#         [mid - width_inner, bottom_limit],   # Bottom Left
#         [mid + width_inner, bottom_limit],   # Bottom Right
#         [mid + width_inner, wing_y],         # Right Notch
#         [mid + width_outer, wing_y]          # Right Wing
#     ], np.int32)

#     # 4. Fill the sharp polygon with white
#     cv2.fillPoly(template, [pts], 255)
    
#     # 5. BLUR: The crucial step for low-res detections
#     # Applying a soft blur (e.g., k=5) makes the template look like image_0.png
#     k_size = int(size * 0.08) # responsive kernel size (around 5 for 60px)
#     if k_size % 2 == 0: k_size += 1 # must be odd
#     soft_template = cv2.GaussianBlur(template, (k_size, k_size), 0)
    
#     return soft_template


def template_from_img(size, path='Models/traffic_light_templates/final_template.jpg'):
    small_img = cv2.imread(path, 0)

    # _, template_mask = cv2.threshold(small_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    width = size
    height = size
    upscaled = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_CUBIC)

    final_template = cv2.GaussianBlur(upscaled, (3, 3), 0)


    return final_template

def create_template(path='Output/cropped_traffic_light_gra.jpg'):

    small_img = cv2.imread(path, 0)

    _, template_mask = cv2.threshold(small_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save or use for matching
    cv2.imwrite('template_wip.jpg', template_mask)
    


# if __name__ == "__main__":
#     im1_name = 'Output/zoomed_traffic_light_single.jpg'
#     # im1_name = 'Output/zoomed_traffic_lights_double.jpg'
#     # im1_name = 'Output/zoomed_red_double.jpg'
#     # temp_name = 'test_crop.jpg'
#     # arrow_from_img(10, temp_name)
    
#     im = cv2.imread(im1_name, cv2.IMREAD_COLOR)

#     classify_light(im, torch.tensor([0, 0, im.shape[1], im.shape[0]]))

#     # create_template()