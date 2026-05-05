"""
语音分离效果评估脚本（无参考音频版）

评估三个维度：
  1. 混合一致性  - 分离后的多路相加是否还原原始混合
  2. 能量分布    - 每路输出的有效能量占比，判断是否有「空轨」
  3. 说话人区分度 - 用 speaker embedding 余弦相似度判断两路是否真的是不同人

如需更准确的 speaker embedding，请在 ASR 环境安装：
    pip install resemblyzer
未安装时会自动回退到 MFCC + GMM 风格的简易特征。
"""
import os
import glob
import argparse
import itertools
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIX_PATH = os.path.join(BASE_DIR, "2in1.wav")
SEP_GLOB = os.path.join(BASE_DIR, "分离_说话人_*.wav")
SEP_GLOB_ENHANCED = os.path.join(BASE_DIR, "分离_增强_说话人_*.wav")
SAMPLE_RATE = 16000


def load_audio(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} 采样率 {sr} != {SAMPLE_RATE}")
    return data.astype(np.float32)


def db(x):
    return 10 * np.log10(x + 1e-12)


# ---------- 1. 混合一致性 ----------
def mix_consistency(mix, sources):
    n = min(len(mix), min(len(s) for s in sources))
    mix = mix[:n]
    sources = [s[:n] for s in sources]
    recon = np.sum(sources, axis=0)

    # 分离模型输出可能整体有缩放，最优缩放后再算残差
    alpha = np.dot(mix, recon) / (np.dot(recon, recon) + 1e-12)
    residual = mix - alpha * recon

    sig_db = db(np.mean(mix ** 2))
    res_db = db(np.mean(residual ** 2))
    snr = sig_db - res_db
    return {
        "最优缩放系数": float(alpha),
        "残差能量(dB)": float(res_db),
        "重建信噪比(dB)": float(snr),
    }


# ---------- 2. 能量分布 ----------
def energy_distribution(sources):
    energies = np.array([np.mean(s ** 2) for s in sources])
    total = energies.sum() + 1e-12
    ratios = energies / total
    return {
        f"轨{i+1}能量占比": float(r) for i, r in enumerate(ratios)
    } | {
        f"轨{i+1}能量(dB)": float(db(e)) for i, e in enumerate(energies)
    }


# ---------- 3. 说话人区分度 ----------
def speaker_embedding_resemblyzer(sources):
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder(verbose=False)
    embs = []
    for s in sources:
        wav = preprocess_wav(s, source_sr=SAMPLE_RATE)
        embs.append(encoder.embed_utterance(wav))
    return np.stack(embs)


def speaker_embedding_mfcc(sources):
    """无 resemblyzer 时的回退方案：MFCC 均值向量"""
    try:
        import librosa
    except ImportError:
        return None
    embs = []
    for s in sources:
        mfcc = librosa.feature.mfcc(y=s, sr=SAMPLE_RATE, n_mfcc=20)
        embs.append(mfcc.mean(axis=1))
    return np.stack(embs)


def cosine_matrix(embs):
    norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


def speaker_separation(sources):
    embs = None
    method = None
    try:
        embs = speaker_embedding_resemblyzer(sources)
        method = "resemblyzer (ECAPA 类)"
    except ImportError:
        embs = speaker_embedding_mfcc(sources)
        method = "MFCC 均值（简易回退，建议 pip install resemblyzer）"
    if embs is None:
        return {"说话人区分度": "未安装 resemblyzer 或 librosa，跳过"}

    sim = cosine_matrix(embs)
    n = len(sources)
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairs[f"轨{i+1} vs 轨{j+1} 余弦相似度"] = float(sim[i, j])
    pairs["特征方法"] = method
    return pairs


# ---------- 4. 有参考时的客观指标（SI-SDR + PIT） ----------
def si_sdr(est, ref, eps=1e-12):
    """Scale-Invariant SDR，单位 dB。数值越大越好，>10 视为良好。"""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.dot(target, target) + eps) /
                         (np.dot(noise, noise) + eps))


def sdr_classic(est, ref, eps=1e-12):
    """普通 SDR（不做缩放不变），便于对比。"""
    noise = est - ref
    return 10 * np.log10((np.dot(ref, ref) + eps) /
                         (np.dot(noise, noise) + eps))


def pit_metrics(estimates, references):
    """
    分离输出和参考的对应关系不固定，遍历所有排列取最大平均 SI-SDR。
    返回最优排列、每路 SI-SDR、平均 SI-SDR、SI-SDR 改善量(SI-SDRi)。
    """
    n = min(len(e) for e in estimates + references)
    estimates = [e[:n] for e in estimates]
    references = [r[:n] for r in references]

    K = len(references)
    if len(estimates) != K:
        raise ValueError(f"估计数({len(estimates)}) != 参考数({K})")

    best = None
    for perm in itertools.permutations(range(K)):
        sisdrs = [si_sdr(estimates[perm[i]], references[i]) for i in range(K)]
        mean = float(np.mean(sisdrs))
        if best is None or mean > best["平均SI-SDR"]:
            best = {
                "排列(估计→参考)": perm,
                "每路SI-SDR(dB)": [float(x) for x in sisdrs],
                "每路SDR(dB)": [
                    float(sdr_classic(estimates[perm[i]], references[i]))
                    for i in range(K)
                ],
                "平均SI-SDR": mean,
            }

    # SI-SDRi: 相对于「直接用 mix 当估计」的提升
    return best


def reference_metrics(mix, estimates, references):
    res = pit_metrics(estimates, references)
    # 基线：把 mix 当作每路估计
    baseline = [si_sdr(mix[:len(r)], r) for r in references]
    res["基线SI-SDR(mix vs ref)"] = [float(x) for x in baseline]
    res["平均SI-SDRi(改善量)"] = float(res["平均SI-SDR"] - np.mean(baseline))
    return res


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="语音分离评估")
    parser.add_argument("--ref", nargs="+", default=None,
                        help="参考音频路径（每个说话人一段干净单轨）。提供后会计算 SI-SDR / SI-SDRi")
    parser.add_argument("--enhanced", action="store_true",
                        help="评估增强版输出（分离_增强_*.wav）")
    args = parser.parse_args()

    sep_paths = sorted(glob.glob(SEP_GLOB_ENHANCED if args.enhanced else SEP_GLOB))
    if not sep_paths:
        raise FileNotFoundError(f"未找到分离结果: {SEP_GLOB}")

    print(f"原始混合: {os.path.basename(MIX_PATH)}")
    print(f"分离轨数: {len(sep_paths)}")
    for p in sep_paths:
        print(f"  - {os.path.basename(p)}")
    print()

    mix = load_audio(MIX_PATH)
    sources = [load_audio(p) for p in sep_paths]

    sections = [
        ("【1】混合一致性（重建信噪比 >10dB 视为良好）", mix_consistency(mix, sources)),
        ("【2】能量分布（占比过低=空轨/弱轨）", energy_distribution(sources)),
        ("【3】说话人区分度（余弦相似度越低分离越彻底）", speaker_separation(sources)),
    ]

    if args.ref:
        refs = [load_audio(p) for p in args.ref]
        ref_res = reference_metrics(mix, sources, refs)
        sections.append(
            ("【4】有参考客观指标（SI-SDR >10dB 良好；SI-SDRi 越大模型贡献越大）", ref_res)
        )

    for title, metrics in sections:
        print(title)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:>20s}: {v:.4f}")
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], float):
                print(f"  {k:>20s}: [" + ", ".join(f"{x:.3f}" for x in v) + "]")
            else:
                print(f"  {k:>20s}: {v}")
        print()

    # 简易判定
    snr = sections[0][1]["重建信噪比(dB)"]
    sim_vals = [v for k, v in sections[2][1].items() if "余弦" in k]
    print("=" * 50)
    print("综合判定:")
    print(f"  - 重建信噪比 {snr:.1f} dB: " +
          ("良好" if snr > 10 else "一般" if snr > 5 else "较差"))
    if sim_vals:
        avg_sim = np.mean(sim_vals)
        print(f"  - 平均说话人相似度 {avg_sim:.3f}: " +
              ("分离彻底" if avg_sim < 0.5 else "有串音" if avg_sim < 0.75 else "疑似同一人/严重串音"))


if __name__ == "__main__":
    main()
