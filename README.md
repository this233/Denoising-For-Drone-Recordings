# 无人机录音智能降噪系统

## 📖 项目简介

本项目是一个基于深度信号处理技术的无人机录音降噪系统，专为解决无人机录制音频中的噪声干扰问题而设计。系统采用多阶段降噪架构，结合机器学习算法和传统信号处理方法，能够有效去除无人机旋翼噪声、电机噪声等各类环境干扰，显著提升音频质量。

### 🎯 核心特性

- **智能噪声库构建** - 基于音频特征自动提取和分类噪声样本
- **多阶段降噪处理** - 频域滤波 → 统计降噪 → 自适应精细降噪 → 残留噪声清理
- **动态噪声匹配** - 使用余弦相似度、欧氏距离和相关系数综合评估最佳匹配噪声
- **长音频分片处理** - 支持任意长度音频的无缝分片处理和重构
- **可视化分析** - 提供详细的频谱分析和降噪效果对比图
- **批量处理能力** - 支持单文件和批量文件处理模式

### 🔬 技术原理

#### 1. 噪声样本提取 (`extract_drone_noise.py`)
- **音频特征提取**：计算RMS能量、频谱质心、频谱带宽、频谱滚降、零交叉率、MFCC系数、频谱对比度
- **变化点检测**：基于滑动窗口统计分析检测音频特征变化点
- **智能分割**：根据变化点和时长约束自动分割音频片段

#### 2. 综合降噪处理 (`drone_denoiser.py`)

**阶段一：频域预处理**
- 高通滤波（50Hz）去除极低频噪声
- 低通滤波（3400Hz）去除高频干扰
- 保留人声关键频率范围（85-3400Hz）

**阶段二：统计降噪**
- 使用 `noisereduce` 库进行基于统计的初步降噪
- 采用平稳噪声假设，降噪强度 80%

**阶段三：自适应精细降噪**
- **动态噪声匹配**：
  ```
  相似度 = 0.4 × 余弦相似度 + 0.3 × 欧氏距离相似度 + 0.3 × 相关系数
  ```
- **频谱减法**：
  ```
  S_clean = max(|S_audio|² - α|S_noise|², β|S_audio|²)
  ```
- **自适应维纳滤波**：
  ```
  H(f) = S_signal(f) / (S_signal(f) + S_noise(f))
  ```

**阶段四：残留噪声后处理**
- 使用残留噪声样本进行最终清理
- 非平稳噪声处理，降噪强度 60%

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 操作系统：Linux/macOS/Windows
- 内存：建议 4GB 以上
- 存储：项目文件约 60MB，环境需要约1GB存储

### 安装步骤

```bash
# 1. 进入项目
cd Denoising-For-Drone-Recordings

# 2. 创建虚拟环境
conda create -p ./env python=3.10 -y
conda activate ./env

# 3. 安装依赖
pip install -r requirements.txt
```

### 依赖包说明

```
numpy>=1.21.0          # 数值计算基础库
scipy>=1.7.0           # 科学计算和信号处理
librosa>=0.9.0         # 音频分析和特征提取
soundfile>=0.10.0      # 音频文件 I/O
matplotlib>=3.4.0      # 数据可视化
seaborn>=0.11.0        # 统计图表绘制
noisereduce>=2.0.0     # 统计降噪算法
scikit-learn>=1.0.0    # 机器学习算法
pyaudio>=0.2.13        # 实时音频流采集
```

## 📚 使用指南

### 基础用法

#### 1. 单文件处理

```bash
python drone_denoiser.py \
    --input ./demo/demo.mp3 \
    --output ./denoised/demo_denoised.mp3 \
    --top-k 10 \
    --similarity-threshold 0.75
```

#### 2. 批量处理

```bash
python drone_denoiser.py \
    --batch \
    --pattern "*.mp3" \
    --input demo \
    --output denoised \
    --top-k 10 \
    --similarity-threshold 0.75
```

#### 3. 噪声样本提取

```bash
# 处理所有文件
python extract_drone_noise.py

# 处理单个文件
python extract_drone_noise.py --file 1.mp3

# 自定义输入输出目录
python extract_drone_noise.py \
    --input /path/to/audio/files \
    --output noise_samples/segments
```

#### 4. 实时音频流（麦克风）

```bash
python drone_denoiser.py \
    --stream \
    --output ./denoised/stream_denoised.wav \
    --stream-rate 44100 \
    --chunk-size 2048 \
    --stream-seconds 60
```

等价的 PyAudio 打开方式如下（内部即采用该配置）：
```python
self.stream = self.audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=44100,
    input=True,
    frames_per_buffer=2048)
```

### 参数说明

#### drone_denoiser.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | - | 输入音频文件或目录路径（非流式） |
| `--output` | str | 必需 | 输出文件或目录路径 |
| `--noise-dir` | str | `noise_samples/segments` | 噪声样本目录 |
| `--batch` | flag | False | 启用批量处理模式 |
| `--pattern` | str | `*.mp3` | 批量处理的文件匹配模式 |
| `--top-k` | int | 5 | 选择前K个最匹配的噪声样本 |
| `--similarity-threshold` | float | 0.75 | 噪声匹配相似度阈值 |
| `--stream` | flag | False | 使用麦克风音频流作为输入 |
| `--stream-rate` | int | 44100 | 麦克风采样率 |
| `--chunk-size` | int | 2048 | 每次读取帧数 |
| `--channels` | int | 1 | 输入声道数（仅支持1） |
| `--device-index` | int | None | 输入设备索引 |
| `--stream-seconds` | int | 60 | 流式处理时长（秒），None 为持续 |

#### extract_drone_noise.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | str | `samples/mp3` | 输入音频文件目录 |
| `--output` | str | `noise_samples/segments` | 噪声输出目录 |
| `--file` | str | None | 处理单个指定文件 |

### 高级配置

#### 1. 自定义降噪参数

```python
from drone_denoiser import DroneVoiceDenoiser

# 创建降噪器实例
denoiser = DroneVoiceDenoiser(
    noise_dir="noise_samples/segments",
    top_k_noise=8,
    similarity_threshold=0.8
)

# 自定义处理参数
denoiser.chunk_duration = 60  # 分片长度（秒）
denoiser.overlap_duration = 10  # 重叠长度（秒）
denoiser.sample_rate = 44100  # 采样率

# 加载噪声样本
denoiser.load_noise_samples()

# 处理音频
result = denoiser.process_file("input.mp3", "output.mp3")
```

#### 2. 自定义噪声提取配置

```python
from extract_drone_noise import DroneNoiseExtractor

# 创建提取器实例
extractor = DroneNoiseExtractor(
    input_dir="audio_files",
    output_dir="noise_samples/segments"
)

# 自定义时间范围
extractor.time_ranges = {
    "recording1.mp3": (30, 180),  # 提取30-180秒片段
    "recording2.mp3": (45, 200),
}

# 自定义分割参数
extractor.max_segment_length = 45  # 最大片段长度
extractor.min_segment_length = 15  # 最小片段长度

# 处理文件
extractor.process_all_files()
```

## 📁 项目结构

```
drone_noise_reduce/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── .gitignore               # Git忽略文件
├── drone_denoiser.py        # 主降噪模块
├── extract_drone_noise.py   # 噪声样本提取模块
├── demo/                    # 演示音频文件
│   └── demo.mp3
├── noise_samples/           # 噪声样本库根目录
│   └── segments/            # 分割后的噪声片段
│       ├── 1_segment_*.mp3
│       ├── 2_segment_*.mp3
│       └── ...
├── denoised/               # 降噪结果输出（自动创建）
└── env/                    # 虚拟环境（自动创建）
```

## 🔧 API 参考

### DroneVoiceDenoiser 类

#### 主要方法

```python
class DroneVoiceDenoiser:
    def __init__(self, noise_dir="noise_samples", top_k_noise=5, similarity_threshold=0.75):
        """初始化降噪器"""
    
    def load_noise_samples(self):
        """加载噪声样本库"""
    
    def process_file(self, input_file, output_file):
        """处理单个音频文件"""
    
    def batch_process(self, input_dir, output_dir, pattern="*.mp3"):
        """批量处理音频文件"""
    
    def multi_stage_denoise(self, y, sr, return_stages=False):
        """多阶段降噪处理核心算法"""

    def process_stream(self, output_file, device_index=None, input_rate=44100, channels=1, chunk_size=2048, stream_seconds=None):
        """实时音频流降噪并写出"""
```

### DroneNoiseExtractor 类

#### 主要方法

```python
class DroneNoiseExtractor:
    def __init__(self, input_dir="samples/mp3", output_dir="samples/extract_noise"):
        """初始化噪声提取器"""
    
    def extract_audio_features(self, y, sr, hop_length=512):
        """提取音频特征"""
    
    def detect_change_points(self, features, window_size=500, threshold_percentile=90):
        """检测音频变化点"""
    
    def process_file(self, filename):
        """处理单个文件"""
    
    def process_all_files(self):
        """处理所有配置的文件"""
```

## 📊 性能与效果

### 处理性能

- **支持格式**：MP3, WAV, FLAC, M4A 等主流音频格式
- **语音清晰度**：显著改善，保持语音自然度
- **噪声抑制**：对旋翼噪声、电机噪声抑制效果显著

## 🛠️ 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保所有依赖已正确安装
   pip install -r requirements.txt --upgrade
   ```

2. **噪声样本不匹配**
   ```python
   # 降低相似度阈值
   denoiser.similarity_threshold = 0.6  # 从0.75降低到0.6
   ```

3. **安装 PyAudio 失败（macOS）**
   ```bash
   # 先安装 PortAudio，再安装 PyAudio
   brew install portaudio
   pip install pyaudio
   ```

### 调试模式

```python
import warnings
warnings.filterwarnings('default')  # 显示警告信息

# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 贡献指南

我们欢迎各种形式的贡献！请按照以下步骤参与项目：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-repo/issues)
- 发送邮件至：your-email@example.com

## 🙏 致谢

感谢以下开源项目的支持：

- [librosa](https://librosa.org/) - 音频分析库
- [noisereduce](https://github.com/timsainb/noisereduce) - 噪声减少算法
- [scipy](https://scipy.org/) - 科学计算库
- [scikit-learn](https://scikit-learn.org/) - 机器学习库

---

*最后更新：2025年7月*