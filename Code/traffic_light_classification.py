import cv2
import numpy as np
import torch

def classify_light(image, box):
    details = {}

    im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    xmin, ymin, xmax, ymax = map(int, box.tolist())
    
    w = (xmax - xmin)
    v = (ymax - ymin)
    aspect_ratio = abs(v/w)

    if aspect_ratio < 1.1:
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

    if qt == 'regular':
        green_start = height - width
        yellow_start = (green_start // 2)

        red_crop = cropped_im[:width]
        green_crop = cropped_im[green_start:]
        yellow_crop = cropped_im[yellow_start: yellow_start + width]

        crops = {"red": red_crop, "yellow": yellow_crop, "green": green_crop}
        crops_list = [crops]

        master_templates = {
            "circle": create_circle_template(width),
            "arrow_up": create_arrow_template(width),
            "arrow_left": cv2.rotate(create_arrow_template(width), cv2.ROTATE_90_COUNTERCLOCKWISE),
            "arrow_right": cv2.rotate(create_arrow_template(width), cv2.ROTATE_90_CLOCKWISE)
        }

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

        crop_left = {"red": red_crop_l, "yellow": yellow_crop_l, "green": green_crop_l}
        crop_right = {"red": red_crop_r, "yellow": yellow_crop_r, "green": green_crop_r}
        crops_list = [crop_left, crop_right]

        master_templates = {
            "circle": create_circle_template(hw),
            "arrow_up": create_arrow_template(hw),
            "arrow_left": cv2.rotate(create_arrow_template(hw), cv2.ROTATE_90_COUNTERCLOCKWISE),
            "arrow_right": cv2.rotate(create_arrow_template(hw), cv2.ROTATE_90_CLOCKWISE)
        }

        cv2.imwrite('Output/test_arrow_up.jpg', create_arrow_template(hw))
        cv2.imwrite('Output/test_arrow_left.jpg', cv2.rotate(create_arrow_template(hw), cv2.ROTATE_90_COUNTERCLOCKWISE))
        cv2.imwrite('Output/test_arrow_right.jpg', cv2.rotate(create_arrow_template(hw), cv2.ROTATE_90_CLOCKWISE))


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
            active_color = max(results, key=lambda k: results[k][1])
            light["color"] = active_color
            light["shape"] = results[active_color][0]
        else:
            light["color"] = "unknown/off"

        details[f"light_{i}"] = light

    return details

    
        

def get_best_match(crop, templates):
    """
    crops: a 2d crop of the light
    templates: a dict {'circle': mask1, 'arrow_up': mask2, ...}
    """
    best_score = -1
    best_label = "off"

    for label, template in templates.items():

        # Perform matching
        res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_label = label

    return best_label, best_score



def create_circle_template(size=25):
    # Create a black square
    template = np.zeros((size, size), dtype=np.uint8)
    # Draw a white filled circle in the center
    center = size // 2
    radius = int(size * 0.33) # Leave a small margin
    cv2.circle(template, (center, center), radius, 255, -1)

    k_size = int(size * 0.08) # responsive kernel size (around 5 for 60px)
    if k_size % 2 == 0: k_size += 1 # must be odd
    soft_template = cv2.GaussianBlur(template, (k_size, k_size), 0)
    
    return soft_template


def create_arrow_template(size=25):
    # 1. Start with a black canvas
    template = np.zeros((size, size), dtype=np.uint8)
    
    # 2. Define "short and fat" geometry as fractions of the 'size'
    # We make the arrow occupy a lot of the width, but keep it short vertically
    mid = size / 2
    top_limit = size * 0.1     # Lowered (making it shorter)
    bottom_limit = size * 0.6   # Raised (making it shorter)
    
    width_outer = size * 0.3    # Wide wings (occupies 90% of width)
    width_inner = size * 0.15    # Wide stem (making it "fat")
    wing_y = size * 0.4         # Point where triangle meets stem

    # 3. Calculate the 7 polygon points
    pts = np.array([
        [mid, top_limit],                    # Tip
        [mid - width_outer, wing_y],         # Left Wing
        [mid - width_inner, wing_y],         # Left Notch
        [mid - width_inner, bottom_limit],   # Bottom Left
        [mid + width_inner, bottom_limit],   # Bottom Right
        [mid + width_inner, wing_y],         # Right Notch
        [mid + width_outer, wing_y]          # Right Wing
    ], np.int32)

    # 4. Fill the sharp polygon with white
    cv2.fillPoly(template, [pts], 255)
    
    # 5. BLUR: The crucial step for low-res detections
    # Applying a soft blur (e.g., k=5) makes the template look like image_0.png
    k_size = int(size * 0.08) # responsive kernel size (around 5 for 60px)
    if k_size % 2 == 0: k_size += 1 # must be odd
    soft_template = cv2.GaussianBlur(template, (k_size, k_size), 0)
    
    return soft_template

# def create_arrow_template(size=25):
#     # Create the 'Master' 25x25 arrow once
#     template = np.zeros((25, 25), dtype=np.uint8)
#     # Scaled coordinates for a 25x25 grid
#     # pts = np.array([[12, 2], [4, 10], [9, 10], [9, 22], 
#     #                 [15, 22], [15, 10], [20, 10]], np.int32)
#     pts = np.array([
#         [12, 5],   # Tip
#         [8, 11],   # Left Wing
#         [11, 11],  # Left Notch
#         [11, 20],  # Bottom Left
#         [14, 20],  # Bottom Right
#         [14, 11],  # Right Notch
#         [17, 11]   # Right Wing
#     ], np.int32)

#     cv2.fillPoly(template, [pts], 255)

#     resized_arrow = cv2.resize(template, (size, size), 
#                                interpolation=cv2.INTER_NEAREST)
    
#     return resized_arrow


if __name__ == "__main__":
    # im1_name = 'Output/zoomed_traffic_light_single.jpg'
    # im1_name = 'Output/zoomed_traffic_lights_double.jpg'
    im1_name = 'Output/zoomed_red_double.jpg'

    im = cv2.imread(im1_name, cv2.IMREAD_COLOR)

    print(classify_light(im, torch.tensor([0, 0, im.shape[1], im.shape[0]])))