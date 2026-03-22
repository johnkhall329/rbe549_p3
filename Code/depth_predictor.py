import cv2
import torch

import matplotlib.pyplot as plt


class DepthPredictor():
    # model_type should be "MiDaS_small", "DPT_Hybrid", or "DPT_Large"
    def __init__(self, model_type="MiDaS_small"):
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)


        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        
    def predict(self, image, format="BGR"):
        if format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            output = prediction.cpu().numpy()

            return output
            # plt.imshow(output)
            # plt.imsave('output.jpg', output)