"""
Ground Truth 标注工具

工作流：
  1. 从指定目录抽 N 条 wav (默认 recordings_cleaned_v2/green++/)
  2. 自动播放每条音频
  3. 显示 ASR 自动转写当起点参考
  4. 你输入"听到的真实文本"，回车保存
  5. 自动保存到 CSV，断电不丢
  6. 重启自动跳过已标注的，继续未标注

快捷键：
  (直接输入)    输入正确文本 + 回车 → 保存并下一条
  /a + 回车     用 ASR 文本一字不改 (ASR 已经对了的常用)
  /r + 回车     重播当前音频
  /s + 回车     跳过这条 (不保存)
  /n <备注>     给这条加备注 (e.g. "/n 噪声大")
  /q + 回车     退出 (已标注的不会丢)
  /b + 回车     提示上一条信息 (需手工改 CSV)

用法：
  python label_groundtruth.py                           默认 100 条 green++
  python label_groundtruth.py --n 50                    抽 50 条
  python label_groundtruth.py --source recordings_cleaned_v2/green+
  python label_groundtruth.py --output my_gt.csv

输出 CSV 列:
  filename, asr_text, correct_text, duration, labeled_at, notes
"""
import argparse
import csv
import glob
import os
import random
import sys
import time

# 音频播放: 优先用 sounddevice (in-process), 失败回退系统播放器
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    try:
        import soundfile as sf
    except ImportError:
        sf = None


def play_audio(path):
    """非阻塞播放音频。"""
    if HAS_SOUNDDEVICE:
        try:
            data, sr = sf.read(path, dtype="float32")
            sd.stop()
            sd.play(data, sr, blocking=False)
            return True
        except Exception as e:
            print(f"   ⚠️ sounddevice 播放失败: {e}, 改用系统播放器")
    # Fallback: 系统默认
    if sys.platform.startswith("win"):
        try:
            os.startfile(path)
        except Exception:
            return False
    elif sys.platform.startswith("darwin"):
        os.system(f"afplay {path!r} &")
    else:
        os.system(f"xdg-open {path!r} &")
    return True


def stop_audio():
    if HAS_SOUNDDEVICE:
        try:
            sd.stop()
        except Exception:
            pass


def get_duration(path):
    """获取音频时长。先 soundfile（快），再 librosa（兼容 MP4-in-WAV 等容器）。"""
    if sf is not None:
        try:
            info = sf.info(path)
            return info.frames / info.samplerate
        except Exception:
            pass  # 落到 librosa fallback
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def load_asr_lookup(csv_path):
    """从 report_v2.csv 加载 filename → asr_text_pf 的映射"""
    if not os.path.exists(csv_path):
        return {}
    out = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                # 兼容多种列名
                asr_text = (row.get("asr_text_pf") or row.get("asr_text")
                            or row.get("text") or "").strip()
                if fname and asr_text:
                    out[fname] = asr_text
    except Exception as e:
        print(f"   ⚠️ 读 ASR 参考失败 ({csv_path}): {e}")
    return out


def load_existing_labels(csv_path):
    """读取已标注的 CSV, 返回 set[filename]"""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("filename", "").strip()
                if fn:
                    done.add(fn)
    except Exception:
        pass
    return done


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Ground Truth 标注工具")
    parser.add_argument("--source", default="recordings_cleaned_v2/green++",
                        help="音频源目录 (默认 green++)")
    parser.add_argument("--n", type=int, default=100,
                        help="目标标注数量 (默认 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="groundtruth.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--asr-csv", default="recordings_cleaned_v2/report_v2.csv",
                        help="ASR 参考 CSV (用于显示参考文本)")
    parser.add_argument("--no-play", action="store_true",
                        help="不自动播放 (用其他工具听)")
    args = parser.parse_args()

    # 路径
    src_dir = args.source if os.path.isabs(args.source) else os.path.join(BASE_DIR, args.source)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(BASE_DIR, args.output)
    asr_csv_path = args.asr_csv if os.path.isabs(args.asr_csv) else os.path.join(BASE_DIR, args.asr_csv)

    if not os.path.isdir(src_dir):
        print(f"❌ 源目录不存在: {src_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    if not files:
        print(f"❌ 找不到 wav: {src_dir}")
        sys.exit(1)

    # 加载 ASR 参考
    asr_lookup = load_asr_lookup(asr_csv_path)
    print(f"📚 加载到 {len(asr_lookup)} 条 ASR 参考文本")

    # 加载已标注
    already_labeled = load_existing_labels(output_path)
    if already_labeled:
        print(f"📌 已有 {len(already_labeled)} 条标注 (在 {os.path.basename(output_path)}), 继续未标注的部分")

    # 抽样: 从未标注的里随机抽 N 条
    random.seed(args.seed)
    candidates = [f for f in files if os.path.basename(f) not in already_labeled]
    if not candidates:
        print(f"✓ 全部 {len(files)} 条都已标注完了")
        sys.exit(0)

    n_remaining_target = args.n - len(already_labeled)
    if n_remaining_target <= 0:
        print(f"✓ 已达到目标 {args.n} 条 (现有 {len(already_labeled)})")
        sys.exit(0)

    random.shuffle(candidates)
    queue = candidates[:n_remaining_target]

    print(f"\n📋 本轮目标: 标 {len(queue)} 条 (已标 {len(already_labeled)} / 目标 {args.n})")
    print()
    print("快捷键:")
    print("  直接输入文本 + Enter   → 保存正确文本并下一条")
    print("  /a + Enter             → 用 ASR 文本不改")
    print("  /r + Enter             → 重播")
    print("  /s + Enter             → 跳过 (不保存)")
    print("  /n <备注> + Enter      → 给这条加备注")
    print("  /q + Enter             → 退出")
    print("=" * 72)

    # CSV (append 模式)
    is_new = not os.path.exists(output_path)
    csv_file = open(output_path, "a", encoding="utf-8-sig", newline="")
    fields = ["filename", "asr_text", "correct_text", "duration", "labeled_at", "notes"]
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if is_new:
        writer.writeheader()
        csv_file.flush()

    last_row = None
    pending_note = ""
    idx = 0
    n_saved = 0

    try:
        while idx < len(queue):
            path = queue[idx]
            fname = os.path.basename(path)
            asr_ref = asr_lookup.get(fname, "")
            dur = get_duration(path)

            # 显示
            print()
            n_total_target = len(already_labeled) + len(queue)
            n_total_done = len(already_labeled) + idx
            pct = (n_total_done / n_total_target * 100) if n_total_target else 0
            print(f"[{idx+1}/{len(queue)} | 累计 {n_total_done+1}/{n_total_target} | {pct:.1f}%]")
            print(f"📁 {fname}  ({dur:.2f}s)")
            if asr_ref:
                print(f"🤖 ASR 参考: {asr_ref}")
            else:
                print(f"🤖 (无 ASR 参考)")

            # 播放
            if not args.no_play:
                play_audio(path)

            # 输入
            try:
                user_input = input("✏️  正确文本: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️ 用户中断")
                break

            # 命令
            if user_input == "/q":
                print("👋 退出 (已标注的已保存)")
                break

            if user_input == "/r":
                print("   🔄 重播")
                continue  # idx 不增

            if user_input == "/s":
                print("   ⏭️ 跳过 (未保存)")
                stop_audio()
                idx += 1
                continue

            if user_input.startswith("/n "):
                pending_note = user_input[3:].strip()
                print(f"   📝 备注待加: {pending_note} (现在请输入正确文本)")
                continue

            if user_input == "/a":
                user_input = asr_ref
                print(f"   → 采用 ASR: {asr_ref}")

            if user_input == "/b":
                if last_row is not None:
                    print(f"   上一条:")
                    print(f"     文件:   {last_row['filename']}")
                    print(f"     ASR:    {last_row['asr_text']}")
                    print(f"     标注:   {last_row['correct_text']}")
                    print(f"   要改的话, 请打开 {os.path.basename(output_path)} 找到该行手改")
                else:
                    print("   没有上一条")
                continue

            # 保存
            row = {
                "filename": fname,
                "asr_text": asr_ref,
                "correct_text": user_input,
                "duration": round(dur, 2),
                "labeled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "notes": pending_note,
            }
            writer.writerow(row)
            csv_file.flush()
            last_row = row
            pending_note = ""
            stop_audio()
            n_saved += 1
            idx += 1

            if not user_input:
                print("   ⚠️ 你输入的是空, 已记录为空字符串 (要修改请手编 CSV)")

    finally:
        csv_file.close()
        stop_audio()

    # 总结
    n_total_now = len(already_labeled) + n_saved
    print()
    print("=" * 72)
    print(f"✓ 本次标注 {n_saved} 条 (累计 {n_total_now} 条)")
    print(f"📁 CSV: {output_path}")
    if n_total_now < args.n:
        remain = args.n - n_total_now
        print(f"💡 还差 {remain} 条达到目标 {args.n}, 直接重跑此脚本会自动续上")
    else:
        print(f"🎉 已达成目标 {args.n} 条, 可以开始用它评估算法 CER 了")


if __name__ == "__main__":
    main()
