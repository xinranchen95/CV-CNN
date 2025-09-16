#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:51:05 2025

@author: chenxinran
"""

from skimage import transform, img_as_ubyte
from PIL import Image, ImageEnhance
import numpy as np
import pytesseract

def x_axil_area (binary_image, gray_image):
    
    horizontal_distribution = np.sum(~binary_image, axis=1)
    max_index_x = np.argmax(horizontal_distribution)
    x_axil_area = gray_image[max_index_x:,:]
    
    return x_axil_area, max_index_x

def x_axil_OCR (x_axil_area):
    
    processed_image_x = img_as_ubyte(x_axil_area)
    pil_image_x = Image.fromarray(processed_image_x)
    width_x, height_x = pil_image_x.size
    new_width_x = int(width_x * 20)
    new_height_x = int(height_x * 20)
    pil_image_x = pil_image_x.resize((new_width_x, new_height_x), Image.BICUBIC)
    enhancer = ImageEnhance.Sharpness(pil_image_x)
    pil_image_x = enhancer.enhance(1.5)
    text_x = pytesseract.image_to_string(pil_image_x)
    text_x = text_x.split('\n')
    
    for n in range(len(text_x)):
        if text_x[n] != '':
            text_x[n] = text_x[n].replace(',', '.')
            x_axil_left_end = text_x[n]
            break
        
    for n in range(1, len(text_x)):
        n = n*(-1)
        if text_x[n] != '':
            text_x[n] = text_x[n].replace(',', '.')
            x_axil_right_end = text_x[n]
            break
    
    return x_axil_left_end, x_axil_right_end

def y_axil_area (binary_image, gray_image):
    
    vertical_distribution = np.sum(~binary_image, axis=0)
    max_index_y = np.argmax(vertical_distribution)
    y_axil_area = gray_image[:,:max_index_y - 10]
    
    return y_axil_area, max_index_y
    
def y_axil_OCR (y_axil_area):
    
    processed_image_y = img_as_ubyte(y_axil_area)
    pil_image_y = Image.fromarray(processed_image_y)
    width_y, height_y = pil_image_y.size
    new_width_y = int(width_y * 2)
    new_height_y = int(height_y * 2)
    pil_image_y = pil_image_y.resize((new_width_y, new_height_y), Image.BICUBIC)
    enhancer = ImageEnhance.Sharpness(pil_image_y)
    pil_image_y = enhancer.enhance(1.5)
    #pil_image_y.show()
    text_y = pytesseract.image_to_string(pil_image_y)
    text_y = text_y.split('\n')

    for n in range(len(text_y)):
        if text_y[n] != '':
            y_axil_up_end = text_y[n]
            break
        
    for n in range(1, len(text_y)):
        n = n*(-1)
        if text_y[n] != '':
            y_axil_down_end = text_y[n]
            break
    
    return y_axil_up_end, y_axil_down_end

def remove_legend (cv_area, cv_area_target):
    
    processed_image_cv = img_as_ubyte(cv_area)
    white_image = Image.new('L', (500, 500))
    white_image.paste(Image.fromarray(processed_image_cv))
    width, height = white_image.size
    new_width = int(width * 10)
    new_height = int(height * 10)
    white_image_p = white_image.resize((new_width, new_height), Image.BICUBIC)
    boxes_info = pytesseract.image_to_boxes(white_image_p)

    X = []
    Y = []
    Text = []
    for box_info in boxes_info.splitlines():
        box_info = box_info.split()
        x, y, w, h = map(int, box_info[1:5])
        text = box_info[0]
        Text.append(text)
        if x != 0 and y != 0:
            X.append(x)
            Y.append(y)   
            print(f"Text: {text}, Position: (x={x}, y={y}, width={w}, height={h})")
    x_max = max(X)
    x_min = min(X)
    y_max = max(Y)
    y_min = min(Y)

    w, h = white_image_p.size
    y_min_real = h - y_max
    y_max_real = h - y_min
    cv_area_target[int(y_min_real/10)-10:int(y_max_real/10),int(x_min/10):int(x_max/10)+20] = 1
    
    return cv_area_target

def standarization (cv_area_target_wo_legend, x_axil_right_end, x_axil_left_end, 
                    y_axil_up_end, y_axil_down_end):
    
    x_range = float(x_axil_right_end) - float(x_axil_left_end)
    y_range = float(y_axil_up_end) - float(y_axil_down_end)
    x_range_pixel = (x_range) * 250
    y_range_pixel = (y_range) * 1
    cv_transform = transform.resize(cv_area_target_wo_legend, (int(y_range_pixel), int(x_range_pixel)))

    white_canvas = np.ones((1000, 1000), dtype=float)

    y_axil_up_end = float(y_axil_up_end)
    location_y = int(abs(500-(y_axil_up_end))*1)
    x_axil_left_end = float(x_axil_left_end)
    location_x = int((x_axil_left_end - (-2))*250)
    white_canvas[location_y:location_y + int(y_range_pixel), location_x:location_x + int(x_range_pixel)] = cv_transform

    white_canvas = (white_canvas * 255).astype(np.uint8)
    white_canvas = Image.fromarray(white_canvas)
    
    return white_canvas
