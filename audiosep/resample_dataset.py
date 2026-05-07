"""
将 dataset/ 下的音频（实际为 MP4 容器，8kHz）批量重采样为 16kHz 单声道 WAV，
输出到 dataset_16k/，文件名保持不变。
"""
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "dataset")
DST_DIR = os.path.join(BASE_DIR, "..", "dataset_16k")
TARGET_SR = 16000


def convert_one(args):
    src, dst = args
    if os.path.exists(dst):
        return src, "skip"
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ar", str(TARGET_SR), "-ac", "1",
        "-c:a", "pcm_s16le",
        "-loglevel", "error",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return src, f"fail: {r.stderr.decode()[:120]}"
    return src, "ok"


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".wav"))
    tasks = [
        (os.path.join(SRC_DIR, f), os.path.join(DST_DIR, f))
        for f in files
    ]
    print(f"待处理: {len(tasks)} 个文件")
    print(f"输出目录: {os.path.abspath(DST_DIR)}")

    t0 = time.time()
    ok = skip = fail = 0
    fail_logs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(convert_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            src, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                fail_logs.append(f"{os.path.basename(src)}: {status}")
            if i % 200 == 0 or i == len(tasks):
                print(f"  进度 {i}/{len(tasks)}  ok={ok} skip={skip} fail={fail}  "
                      f"耗时 {time.time()-t0:.1f}s")

    print(f"\n完成: ok={ok} skip={skip} fail={fail}  总耗时 {time.time()-t0:.1f}s")
    if fail_logs:
        print(f"\n失败示例（最多20条）:")
        for line in fail_logs[:20]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
