# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Modified detect.py with translation + regional TTS support (gTTS fallback to pyttsx3).
Saves minimal temp files and plays audio in a background thread so detection isn't blocked.

Run:
    python detect_with_translation.py --weights runs/train/exp/weights/best.pt --source 0 --view-img

Install required packages:
    pip install googletrans==4.0.0-rc1 gTTS playsound pyttsx3

Note: playsound may need extra system codecs on some platforms; pyttsx3 is the offline fallback.
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import threading
import uuid
import tempfile
import time

# translation & tts
try:
    from googletrans import Translator
except Exception:
    Translator = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None
try:
    import playsound
except Exception:
    playsound = None
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

import torch
import pathlib
import sys

# Fix PosixPath error on Windows
pathlib.PosixPath = pathlib.WindowsPath


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


# ------------------- Helper: audio playback -------------------
def _speak_with_gtts(text, lang):
    """Convert text to speech using gTTS and play (blocking)."""
    if gTTS is None or playsound is None:
        raise RuntimeError("gTTS or playsound not available")
    tf = tempfile.gettempdir()
    fname = os.path.join(tf, f"tts_{uuid.uuid4().hex}.mp3")
    try:
        tts = gTTS(text=str(text), lang=lang)
        tts.save(fname)
        playsound.playsound(fname)
    finally:
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except Exception:
            pass


def _speak_with_pyttsx(text):
    """Synchronous offline TTS via pyttsx3."""
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 not available")
    engine = pyttsx3.init()
    engine.say(str(text))
    engine.runAndWait()


def speak_text_async(text, lang_code="en", prefer_online=True):
    """Play `text` in background thread. Tries gTTS+playsound first (if prefer_online True), otherwise falls back to pyttsx3.

    This function returns immediately; playback is done in a daemon thread so it won't block program exit.
    """
    def worker(txt, lang):
        # Try gTTS first if available and preferred
        if prefer_online and gTTS is not None and playsound is not None:
            try:
                _speak_with_gtts(txt, lang)
                return
            except Exception:
                pass
        # Fallback to local pyttsx3
        if pyttsx3 is not None:
            try:
                _speak_with_pyttsx(txt)
                return
            except Exception:
                pass
        # If nothing works, silently fail (but print)
        print("[TTS] No available TTS engine to speak:", txt)

    t = threading.Thread(target=worker, args=(text, lang_code), daemon=True)
    t.start()


# ------------------- Main run function (modified detect) -------------------
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    # translation/tts options (can be overridden via kwargs with run(...))
    target_language='ml',  # default regional language: Malayalam ('ml')
    tts_prefer_online=True,  # try gTTS first
    tts_min_interval=1.2,  # minimum seconds between spoken items to avoid overlap
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Prepare translator (if available)
    translator = Translator() if Translator is not None else None

    # State used for throttling speech
    prev_label = None
    last_spoken_time = 0.0

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV helper
        csv_path = save_dir / "predictions.csv"

        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        if save_format == 0:
                            coords = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # Annotate image
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label_text = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label_text, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                    # ---------------------- TRANSLATE & TTS ----------------------
                    try:
                        now = time.time()
                        # speak only when label changes and min interval passed
                        if label != prev_label and (now - last_spoken_time) >= tts_min_interval:
                            prev_label = label
                            last_spoken_time = now

                            # get translation (if translator available)
                            translated_text = label
                            if translator is not None:
                                try:
                                    translated_text = translator.translate(label, dest=target_language).text
                                except Exception:
                                    # translation failed, keep original
                                    translated_text = label

                            # speak in background
                            speak_text_async(translated_text, lang_code=target_language, prefer_online=tts_prefer_online)

                    except Exception as e:
                        print("[Translation/TTS] error:", e)
                    # -------------------------------------------------------------

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


# ------------------- CLI parse and main -------------------

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="0=YOLO,1=Pascal-VOC")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # translation CLI args
    parser.add_argument("--target-lang", type=str, default='ml', help="target language code for translation (eg 'hi','ta','ml')")
    parser.add_argument("--tts-online", action="store_true", help="prefer online TTS (gTTS) over offline pyttsx3")
    parser.add_argument("--tts-interval", type=float, default=1.2, help="minimum seconds between spoken phrases")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    opt_dict = vars(opt).copy()
    target_lang = opt_dict.pop("target_lang", "ml")
    tts_online = opt_dict.pop("tts_online", False)
    tts_interval = opt_dict.pop("tts_interval", 1.2)

    run(
        **opt_dict,
        target_language=target_lang,
        tts_prefer_online=tts_online,
        tts_min_interval=tts_interval,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
