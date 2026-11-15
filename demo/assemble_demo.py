#!/usr/bin/env python3
"""Assemble the FashionShift demo video: app screenshot pans + narration + burned captions."""
import subprocess, os, sys

D = os.path.dirname(os.path.abspath(__file__))
# (shot, pan direction, duration, [caption chunks])
SEGS = [
    ("bench1.png", "down", 18.76, [
        "Clothing photos look simple to classify.",
        "But does a model that scores well on clean test images survive when the picture is shifted?",
        "FashionShift is a workbench for testing exactly that.",
        "It trains three models on Fashion-MNIST, then audits them on a test set they never saw during training.",
    ]),
    ("bench2.png", "up", 21.36, [
        "On the live bench, every image comes from the untouched official test set.",
        "The top line is the dataset truth.",
        "Below it are a logistic regression, a small convolutional network, and the same network trained with augmentation.",
        "Raise the abstain threshold and low-confidence calls go gray.",
        "All of this runs in your browser.",
    ]),
    ("evidence.png", "down", 21.28, [
        "The evidence tab keeps it honest.",
        "On the untouched 10,000-image test set, the plain network scores 91%.",
        "The augmented network trades a little clean accuracy, landing at 88.5%.",
        "And the weak classes are the same in every model: shirts, pullovers, and tees blur together.",
    ]),
    ("stress.png", "down", 23.20, [
        "The stress lab is where the models break.",
        "Blur, occlude, or shift the images and watch the calls change live.",
        "A four-pixel shift collapses logistic regression to 41%, and drops the plain network to 58%.",
        "The augmented network holds 80%.",
        "But under blur and occlusion, augmentation does not help at all.",
    ]),
    ("stress.png", "up", 15.04, [
        "The uncomfortable part: confidence stays high while accuracy falls.",
        "Even on shifted images, wrong calls average 60 to 75% confidence.",
        "So abstaining on confidence alone will not catch shifted inputs.",
    ]),
    ("import.png", "down", 15.96, [
        "You can import your own image or a spreadsheet batch.",
        "The pipeline shows exactly what the model sees. Sessions stay in your browser, and the model card exports with one click.",
        "Honest numbers, honest limits.",
        "That is FashionShift.",
    ]),
]
W, H = 1280, 720


def tsec(s):
    m = int(s // 60); sec = s - 60 * m
    return f"0:{m:02d}:{sec:05.2f}"


def content_bottom(src):
    """Last row whose pixels differ meaningfully from the page background."""
    from PIL import Image
    import numpy as np
    im = np.asarray(Image.open(src).convert("RGB"), dtype=np.int16)
    bg = np.array([13, 17, 23])
    diff = np.abs(im - bg).sum(axis=2).max(axis=1)
    rows = np.where(diff > 40)[0]
    return int(rows.max()) + 20 if len(rows) else im.shape[0]


def main():
    # 1) per-segment pan clips
    clips = []
    for i, (shot, direction, dur, _) in enumerate(SEGS, 1):
        src = os.path.join(D, "shots", shot)
        cb = content_bottom(src)
        print(f"{shot}: content bottom {cb}")
        out = os.path.join(D, f"clip{i}.mp4")
        rng = max(0, cb - H)
        yexpr = f"{rng}*t/{dur}" if direction == "down" else f"{rng}*(1-t/{dur})"
        vf = f"crop={W}:{H}:0:'{yexpr}',fps=30,format=yuv420p"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", src,
                        "-vf", vf, "-t", str(dur), "-c:v", "libx264", "-preset", "fast",
                        "-crf", "19", out], check=True)
        clips.append(out)

    # 2) concat video
    with open(os.path.join(D, "concat.txt"), "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    vcat = os.path.join(D, "video_raw.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", os.path.join(D, "concat.txt"), "-c", "copy", vcat], check=True)

    # 3) concat audio
    with open(os.path.join(D, "aconcat.txt"), "w") as f:
        for i in range(1, 7):
            f.write(f"file '{os.path.join(D, f'seg{i}.wav')}'\n")
    acat = os.path.join(D, "narration.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", os.path.join(D, "aconcat.txt"), "-c", "copy", acat], check=True)

    # 4) captions: ASS + SRT
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
           "[V4+ Styles]",
           "Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,Bold,Italic,"
           "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
           "MarginL,MarginR,MarginV,Encoding",
           "Style: Cap,DejaVu Sans,22,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,3,0,0,2,48,48,30,1",
           "", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"]
    srt = []
    t = 0.0
    k = 0
    for _, _, dur, chunks in SEGS:
        per = dur / len(chunks)
        for c in chunks:
            k += 1
            a, b = t + 0.15, t + per - 0.12
            ass.append(f"Dialogue: 0,{tsec(a)},{tsec(b)},Cap,,0,0,0,,{c}")
            srt.append(f"{k}\n{int(a//3600):02d}:{int(a//60)%60:02d}:{a%60:06.3f} --> "
                       f"{int(b//3600):02d}:{int(b//60)%60:02d}:{b%60:06.3f}\n{c}\n".replace(".", ","))
            t += per
    with open(os.path.join(D, "captions.ass"), "w") as f:
        f.write("\n".join(ass))
    with open(os.path.join(D, "captions.srt"), "w") as f:
        f.write("\n".join(srt))

    # 5) mux + burn captions
    out = os.path.join(D, "FashionShift_Demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vcat, "-i", acat,
                    "-vf", f"subtitles={os.path.join(D, 'captions.ass')}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", out], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", out], capture_output=True, text=True)
    print("demo video:", out, "duration", r.stdout.strip())


if __name__ == "__main__":
    main()
