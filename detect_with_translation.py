
# detect_with_translation.py
# Ultralytics 🚀 - Modified YOLOv5 detect for live translation + TTS

import argparse
import os
import sys
import threading
import tempfile
import uuid
import time
from pathlib import Path
import pathlib
import os
import sys

# Fix PosixPath error on Windows
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath


import torch
import cv2

# Translation & TTS
try:
    from googletrans import Translator
except ImportError:
    Translator = None
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
try:
    import playsound
except ImportError:
    playsound = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# YOLOv5 imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_boxes,
                           xyxy2xywh, increment_path, check_requirements)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

# ------------------ Helper functions ------------------

def _speak_with_gtts(text, lang):
    """Use gTTS + playsound"""
    if gTTS is None or playsound is None:
        raise RuntimeError("gTTS or playsound not available")
    tf = tempfile.gettempdir()
    fname = os.path.join(tf, f"tts_{uuid.uuid4().hex}.mp3")
    try:
        tts = gTTS(text=str(text), lang=lang)
        tts.save(fname)
        playsound.playsound(fname)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

def _speak_with_pyttsx(text):
    """Offline TTS using pyttsx3"""
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 not available")
    engine = pyttsx3.init()
    engine.say(str(text))
    engine.runAndWait()

def speak_text_async(text, lang_code="en", prefer_online=True):
    """Speak in background thread"""
    def worker(txt, lang):
        if prefer_online and gTTS is not None and playsound is not None:
            try:
                _speak_with_gtts(txt, lang)
                return
            except Exception:
                pass
        if pyttsx3 is not None:
            try:
                _speak_with_pyttsx(txt)
                return
            except Exception:
                pass
        print("[TTS] No TTS engine available:", txt)
    t = threading.Thread(target=worker, args=(text, lang_code), daemon=True)
    t.start()

# ------------------ Main Detection ------------------

@smart_inference_mode()
def run(weights=ROOT / "yolov5s.pt",
        source=0,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        device="",
        view_img=True,
        target_language="ml",
        tts_prefer_online=True,
        tts_min_interval=1.2):

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    translator = Translator() if Translator is not None else None

    prev_label = None
    last_spoken_time = 0.0

    source = str(source)
    webcam = source.isnumeric() or source.endswith(".streams")
    dataset = LoadStreams(source, img_size=imgsz, stride=stride) if webcam else LoadImages(source, img_size=imgsz, stride=stride)

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s.copy()
            annotator = Annotator(im0, line_width=3, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c]
                    label_text = f"{label} {conf:.2f}"
                    annotator.box_label(xyxy, label_text, color=colors(c, True))

                    now = time.time()
                    if label != prev_label and (now - last_spoken_time) >= tts_min_interval:
                        prev_label = label
                        last_spoken_time = now
                        translated_text = label
                        if translator is not None:
                            try:
                                translated_text = translator.translate(label, dest=target_language).text
                            except Exception:
                                pass
                        speak_text_async(translated_text, lang_code=target_language, prefer_online=tts_prefer_online)

            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(path), im0)
                if cv2.waitKey(1) == ord('q'):
                    return

# ------------------ CLI ------------------

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--target-lang", type=str, default="ml")
    parser.add_argument("--tts-online", action="store_true")
    parser.add_argument("--tts-interval", type=float, default=1.2)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(weights=opt.weights,
        source=opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        view_img=opt.view_img,
        target_language=opt.target_lang,
        tts_prefer_online=opt.tts_online,
        tts_min_interval=opt.tts_interval)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
