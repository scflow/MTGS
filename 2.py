import pickle
from collections import Counter, defaultdict
from pathlib import Path
import math

pkl = Path("data/main_mt/MTGS/road_block-331220_4690660_331190_4690710/video_scene_dict.pkl")
data = pickle.load(open(pkl, "rb"))

travel_id = 0
frame_start = 0
frame_end = None  # None=到最后

# 找对应的 video_token
video_token = next(vt for vt in data if vt.endswith(f"-{travel_id}"))
frames = data[video_token]["frame_infos"]
if frame_end is None:
    frame_end = len(frames) - 1

stats = defaultdict(lambda: {"min": 1e9, "sum": 0.0, "count": 0, "min_frame": None})
nearest_counts = Counter()

for idx, f in enumerate(frames[frame_start:frame_end + 1], start=frame_start):
    best = None
    for name, token, box in zip(f["gt_names"], f["track_tokens"], f["gt_boxes"]):
        if name != "vehicle":
            continue
        x, y, z = box[:3]
        if x <= 0:  # 只看前方车辆；不需要可删
            continue
        d = math.sqrt(x * x + y * y + z * z)
        s = stats[token]
        s["sum"] += d
        s["count"] += 1
        if d < s["min"]:
            s["min"] = d
            s["min_frame"] = idx
        if best is None or d < best[0]:
            best = (d, token)
    if best:
        nearest_counts[best[1]] += 1

# 按“最近次数最多”排序
print("nearest_counts top:")
for token, cnt in nearest_counts.most_common(10):
    s = stats[token]
    avg = s["sum"] / s["count"]
    print(token, "nearest_frames", cnt, "avg_dist", round(avg, 2), "min_dist", round(s["min"], 2), "min_frame", s["min_frame"])
