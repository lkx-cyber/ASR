import os
import onnxruntime as ort
import numpy as np
import soundfile as sf

# ===================== 配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.onnx")  # 本地模型
INPUT_WAV = os.path.join(BASE_DIR, "2in1.wav")                    # 输入音频
OUTPUT_DIR = BASE_DIR                                              # 输出目录
SAMPLE_RATE = 16000             # 模型固定16k
# =================================================

# 加载模型
print("正在加载本地模型...")
session = ort.InferenceSession(MODEL_PATH)

# 查看模型真实输入名（自动校验）
print("模型输入名：", [i.name for i in session.get_inputs()])

# 读取音频（自动转16k单声道）
def load_audio(path):
    data, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError("必须是16000Hz音频")
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # 转单声道
    return data.astype(np.float32)

# 保存音频
def save(path, data):
    sf.write(path, data, SAMPLE_RATE)
    print(f"已保存：{path}")

# 主程序
if __name__ == "__main__":
    mix = load_audio(INPUT_WAV)
    print(f"音频长度：{len(mix)} 采样点")

    # 模型推理 —— 这里已修复！输入名改成 inputs
    print("\n开始分离多个人声...")
    input_tensor = np.expand_dims(mix, 0)
    
 
    outputs = session.run(None, {"inputs": input_tensor})  


    # 在当前上当下输出多个人声，每人一个文件
    for idx, speaker in enumerate(outputs):
        speaker_data = speaker.squeeze()
        save(os.path.join(OUTPUT_DIR, f"分离_说话人_{idx+1}.wav"), speaker_data)

    print("\n多说话人分离完成！")
    print(f"共分离出 {len(outputs)} 个不同人的声音！")