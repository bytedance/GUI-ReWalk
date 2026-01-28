# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .utils import get_yolo_model, detect_gui_from_img, draw_bbox
from PIL import Image
import io

# ***************************
model_path='omniparser/weights/icon_detect/model.pt'
BOX_TRESHOLD = 0.05
# ***************************

class GUIDecorator:
    def __init__(self, model_path, BOX_TRESHOLD = 0.05, device = 'cpu', yolo_print=False):
        self.model_path = model_path
        self.BOX_TRESHOLD = BOX_TRESHOLD
        self.device = device
        self.model = get_yolo_model(model_path)
        self.model.to(device)
        self.yolo_print = yolo_print

    def detect_gui(self, image_ori, text=[], ocr_bbox=[]):
        if isinstance(image_ori, bytes):
            image = Image.open(io.BytesIO(image_ori))
        elif isinstance(image_ori, str):
            image = Image.open(image_ori)
        else:
            image = image_ori
        
        image_rgb = image.convert('RGB')
        boxes, logits = detect_gui_from_img(image_rgb, self.model, BOX_TRESHOLD = self.BOX_TRESHOLD, ocr_bbox=ocr_bbox, 
                                        ocr_text=text, iou_threshold=0.7, scale_img=False, batch_size=1, yolo_print=self.yolo_print)
        return boxes, logits
    
    def draw_bbox(self, image_ori, boxes, logits, format='array'):
        if isinstance(image_ori, bytes):
            image = Image.open(io.BytesIO(image_ori))
        elif isinstance(image_ori, str):
            image = Image.open(image_ori)
        else:
            image = image_ori
        
        image_rgb = image.convert('RGB')
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        image_rgb, _ = draw_bbox(image_rgb, boxes, logits, draw_bbox_config=draw_bbox_config, format=format)
        return image_rgb