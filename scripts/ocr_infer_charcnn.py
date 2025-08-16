#!/usr/bin/env python3
import os
import json
import argparse
from typing import List

import cv2
import numpy as np
import torch

from models.ocr.char_cnn import load_charcnn


def segment_chars(gray: np.ndarray) -> List[np.ndarray]:
    h, w = gray.shape
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 3), np.uint8)
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= 0.4 * gray.shape[0] and w >= 5:
            rois.append(gray[y:y + h, x:x + w])
    return rois


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='models/ocr_charcnn/charcnn_best.pt')
    parser.add_argument('--classes', default='models/ocr_charcnn/classes.json')
    args = parser.parse_args()

    model, classes = load_charcnn(args.model, args.classes)
    device = next(model.parameters()).device

    bgr = cv2.imread(args.image)
    if bgr is None:
        print('image not found')
        return
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rois = segment_chars(gray)
    transform = lambda im: torch.from_numpy(cv2.resize(im, (32, 32))).unsqueeze(0).unsqueeze(0).float().div(255.0).sub(0.5).div(0.5)
    preds = []
    with torch.no_grad():
        for r in rois:
            x = transform(r).to(device)
            logits = model(x)
            idx = int(logits.argmax(dim=1).item())
            preds.append(classes[idx])
    print(''.join(preds))


if __name__ == '__main__':
    main()



