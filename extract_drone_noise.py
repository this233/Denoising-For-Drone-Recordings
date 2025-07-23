#!/usr/bin/env python3
"""
无人机录音噪声提取与智能分割系统

本模块实现了一个基于音频特征分析的智能噪声提取系统，专门用于
从无人机录音中提取和分类噪声样本。系统采用多维特征提取和变化点
检测算法，能够自动识别音频中的噪声片段并进行智能分割。

核心技术特点：
- 多维音频特征提取：RMS能量、频谱质心、频谱带宽、频谱滚降、
  零交叉率、MFCC系数、频谱对比度等
- 智能变化点检测：基于滑动窗口统计分析的音频特征变化检测
- 自适应分割策略：根据变化点和时长约束自动分割噪声片段
- 可视化分析：提供详细的特征变化和分割效果可视化

应用场景：
- 构建无人机噪声样本库
- 音频噪声特征分析
- 噪声片段自动标注
- 声学数据预处理

作者：[项目作者]
版本：v1.0.0
创建日期：2025年
最后修改：2025年7月

使用示例：
    # 基础用法
    extractor = DroneNoiseExtractor()
    extractor.process_all_files()
    
    # 自定义配置
    extractor = DroneNoiseExtractor(
        input_dir="audio_files",
        output_dir="extracted_noise"
    )
    extractor.process_file("drone_recording.mp3")
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 配置matplotlib中文字体支持
# 确保图表中的中文标签能够正确显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DroneNoiseExtractor:
    """
    无人机噪声智能提取器
    
    本类实现了一个完整的无人机噪声提取和分析系统，采用基于音频特征
    变化的智能分割算法，能够从长时间录音中自动提取出高质量的噪声片段。
    
    核心算法原理：
    
    1. 多维特征提取：
       - RMS能量：反映音频整体能量水平
       - 频谱质心：反映频谱能量的重心位置
       - 频谱带宽：反映频率分布的集中程度
       - 频谱滚降：反映高频能量分布
       - 零交叉率：反映音频的周期性特征
       - MFCC系数：反映音频的倒谱特征
       - 频谱对比度：反映不同频段的能量对比
    
    2. 变化点检测算法：
       - 滑动窗口统计分析：计算特征在时间窗口内的变化
       - 方差阈值判断：识别显著的特征变化点
       - 峰值检测：定位变化最剧烈的时间点
    
    3. 智能分割策略：
       - 基于变化点的自然分割
       - 时长约束优化（最小/最大长度限制）
       - 音频质量评估和过滤
    
    Attributes:
        input_dir (Path): 输入音频文件目录路径
        output_dir (Path): 噪声片段输出目录路径
        time_ranges (dict): 预配置的文件时间提取范围
        max_segment_length (int): 最大片段长度限制（秒）
        min_segment_length (int): 最小片段长度限制（秒）
    
    Notes:
        - 系统支持MP3、WAV、FLAC等主流音频格式
        - 所有提取的片段都会保存为MP3格式以节省存储空间
        - 变化点检测算法适用于各种类型的无人机噪声
        - 输出文件采用标准化命名格式便于后续处理
    
    Examples:
        >>> extractor = DroneNoiseExtractor("recordings", "noise_output")
        >>> extractor.process_file("drone_flight.mp3")
        >>> extractor.process_all_files()
    """
    
    def __init__(self, input_dir="samples/mp3", output_dir="noise_samples/segments"):
        """
        初始化无人机噪声提取器
        
        设置输入输出目录，配置默认的时间范围和分割参数。
        
        Args:
            input_dir (str): 输入音频文件目录路径。
                           应包含需要提取噪声的无人机录音文件。
            output_dir (str): 噪声片段输出目录路径。
                            系统会自动创建该目录（如果不存在）。
        
        Note:
            - 输出目录会自动创建，包括必要的子目录结构
            - 时间范围配置可以通过修改time_ranges属性进行调整
            - 分割参数可以根据实际需求进行优化
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # === 预配置的文件时间提取范围 ===
        # 每个文件的(开始时间, 结束时间)，单位：秒
        # 这些时间范围是基于实际录音内容分析得出的最佳噪声片段
        self.time_ranges = {
            "1.mp3": (90, 510),
            "2.mp3": (15, 360),
            "3.mp3": (30, 495),
            "4.mp3": (60, 540),
            "5.mp3": (40, 480),
            "6.mp3": (35, 500),
            "7.mp3": (50, 590),
            "8.mp3": (50, 570),
            "9.mp3": (45, 525),
            "10.mp3": (25, 410)
        }
        
        # === 音频分割参数配置 ===
        self.max_segment_length = 30  # 最大片段长度（秒），避免单个文件过大
        self.min_segment_length = 10  # 最小片段长度（秒），确保样本有效性
        
    def extract_audio_features(self, y, sr, hop_length=512):
        """
        多维音频特征提取算法
        
        提取多种音频特征用于后续的变化点检测和音频分析。
        每种特征都反映了音频信号的不同方面特性，组合使用可以
        全面描述音频的时频域特征变化。
        
        提取的特征包括：
        1. RMS能量：Root Mean Square，反映音频整体能量水平
        2. 频谱质心：Spectral Centroid，反映频谱能量重心位置
        3. 频谱带宽：Spectral Bandwidth，反映频率分布的集中程度
        4. 频谱滚降：Spectral Rolloff，反映高频能量分布特征
        5. 零交叉率：Zero Crossing Rate，反映音频周期性和噪声特征
        6. MFCC系数：Mel-frequency Cepstral Coefficients，反映音色特征
        7. 频谱对比度：Spectral Contrast，反映不同频段能量对比
        
        Args:
            y (np.ndarray): 输入音频信号，形状为(n_samples,)
            sr (int): 音频采样率
            hop_length (int): STFT变换的帧移长度，影响时间分辨率
        
        Returns:
            np.ndarray: 特征矩阵，形状为(n_features, n_frames)
                       每一列代表一个时间帧的特征向量
        
        Note:
            - 所有特征都使用相同的hop_length以确保时间对齐
            - 特征矩阵经过转置，每行代表一个时间帧
            - MFCC只取前5个系数以减少计算复杂度
            - 频谱对比度使用默认的频段划分
        """
        # 计算帧长度
        frame_length = hop_length * 2
        
        # 1. 能量/功率
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # 2. 频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # 3. 频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        
        # 4. 频谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        
        # 5. 零交叉率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        
        # 6. MFCC前几个系数
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5, hop_length=hop_length)
        
        # 7. 频谱对比度
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        
        # 组合特征
        features = np.vstack([
            energy,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing_rate,
            mfcc,
            spectral_contrast
        ])
        
        return features.T  # 转置，每行是一个时间帧的特征向量
    
    def detect_change_points(self, features, window_size=500, threshold_percentile=90):
        """
        智能变化点检测算法
        
        基于滑动窗口统计分析的音频特征变化点检测算法。通过分析
        相邻时间窗口内特征统计量的差异，识别音频中的显著变化点，
        这些变化点通常对应于不同的飞行状态、噪声类型转换等。
        
        算法流程：
        1. 特征标准化：对所有特征进行Z-score标准化
        2. 滑动窗口分析：
           - 对每个时间点，计算前后窗口的特征统计量
           - 使用欧氏距离量化前后窗口的差异程度
        3. 变化强度计算：
           - change_score[i] = ||mean_after - mean_before||₂
           - 距离越大表示变化越显著
        4. 峰值检测：
           - 使用percentile阈值过滤弱变化点
           - 应用find_peaks算法定位局部最大值
        5. 边界处理：确保变化点不会导致过短或过长的片段
        
        Args:
            features (np.ndarray): 音频特征矩阵，形状为(n_frames, n_features)
            window_size (int): 滑动窗口大小（时间帧数）
                              较大的窗口对长期变化敏感，较小的窗口对短期变化敏感
            threshold_percentile (float): 变化强度阈值百分位数
                                        更高的百分位数会检测到更少但更显著的变化点
        
        Returns:
            tuple: (change_points, change_scores)
                - change_points (np.ndarray): 变化点位置数组（时间帧索引）
                - change_scores (np.ndarray): 每个时间帧的变化强度得分
        
        Note:
            - 窗口大小应该根据音频帧率和期望的时间分辨率调整
            - 阈值百分位数影响变化点的数量，需要根据实际需求平衡
            - 算法会自动排除边界区域以避免边界效应
            - 变化点检测对噪声比较鲁棒，适用于各种无人机录音环境
        """
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 计算滑动窗口内的特征方差
        n_frames = len(features_scaled)
        change_scores = []
        print(f"n_frames: {n_frames},window_size: {window_size}")
        for i in range(window_size, n_frames - window_size):
            # 前窗口和后窗口的特征
            prev_window = features_scaled[i-window_size:i]
            next_window = features_scaled[i:i+window_size]
            
            # 计算两个窗口的特征差异
            prev_mean = np.mean(prev_window, axis=0)
            next_mean = np.mean(next_window, axis=0)
            
            # 欧氏距离作为变化度量
            change_score = np.linalg.norm(next_mean - prev_mean)
            change_scores.append(change_score)
        
        change_scores = np.array(change_scores)
        
        # 找到变化点
        threshold = np.percentile(change_scores, threshold_percentile)
        change_points = find_peaks(change_scores, height=threshold, distance=window_size//2)[0]
        
        # 调整索引（因为我们从window_size开始计算）
        change_points = change_points + window_size
        
        return change_points, change_scores
    
    def segment_audio(self, y, sr, change_points, start_time=0):
        """
        根据变化点分割音频
        """
        segments = []
        
        # 转换变化点为时间
        change_times = librosa.frames_to_time(change_points, sr=sr)
        
        # 添加起始和结束时间
        all_times = [0] + list(change_times) + [len(y) / sr]
        print(f"all_times: {all_times}")
        
        for i in range(len(all_times) - 1):
            segment_start = all_times[i]
            segment_end = all_times[i + 1]
            segment_duration = segment_end - segment_start
            
            # 如果片段太长，进一步分割
            if segment_duration > self.max_segment_length:
                # 按最大长度分割
                current_start = segment_start
                while current_start < segment_end:
                    current_end = min(current_start + self.max_segment_length, segment_end)
                    if current_end - current_start >= self.min_segment_length:
                        start_sample = int(current_start * sr)
                        end_sample = int(current_end * sr)
                        segment_audio = y[start_sample:end_sample]
                        segments.append({
                            'audio': segment_audio,
                            'start_time': start_time + current_start,
                            'end_time': start_time + current_end,
                            'duration': current_end - current_start
                        })
                    current_start = current_end
            elif segment_duration >= self.min_segment_length:
                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                segment_audio = y[start_sample:end_sample]
                segments.append({
                    'audio': segment_audio,
                    'start_time': start_time + segment_start,
                    'end_time': start_time + segment_end,
                    'duration': segment_duration
                })
        
        return segments
    
    def process_file(self, filename):
        """
        处理单个音频文件的完整流程
        
        对指定的音频文件执行完整的噪声提取和分割流程，包括：
        音频加载、特征提取、变化点检测、智能分割和文件保存。
        
        处理流程：
        1. 验证文件存在性和配置有效性
        2. 加载音频文件并提取指定时间段
        3. 提取多维音频特征
        4. 执行变化点检测算法
        5. 基于变化点进行智能分割
        6. 保存分割后的音频片段
        7. 生成可视化分析图表
        
        Args:
            filename (str): 要处理的音频文件名
                          必须存在于input_dir中且在time_ranges中有配置
        
        Note:
            - 文件必须在预配置的time_ranges中有时间范围设置
            - 输出文件采用标准命名格式：{drone_id}_segment_{seq}_{start}s-{end}s.mp3
            - 处理过程会生成详细的控制台输出用于监控进度
            - 如果检测到变化点，会生成可视化分析图表
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            print(f"警告: 文件 {filename} 不存在")
            return
        
        if filename not in self.time_ranges:
            print(f"警告: 文件 {filename} 没有配置时间范围")
            return
        
        start_time, end_time = self.time_ranges[filename]
        
        print(f"正在处理文件: {filename}")
        print(f"提取时间段: {start_time}s - {end_time}s")
        
        # 加载音频文件
        y, sr = librosa.load(file_path, sr=None)
        
        print(f"sr(sample rate): {sr},sample length: {int(end_time * sr)-int(start_time * sr)}")
        # 提取指定时间段
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_segment = y[start_sample:end_sample]
        
        print(f"提取音频长度: {len(y_segment)/sr:.1f}s")
        
        # 提取音频特征
        features = self.extract_audio_features(y_segment, sr)
        
        # 检测变化点
        window_size = 500  # 与detect_change_points中的默认值保持一致
        change_points, change_scores = self.detect_change_points(features, window_size=window_size)
        
        print(f"检测到 {len(change_points)} 个变化点")
        
        # 分割音频
        segments = self.segment_audio(y_segment, sr, change_points, start_time)
        
        print(f"分割得到 {len(segments)} 个片段")
        
        # 保存片段
        base_name = filename.split('.')[0]
        for i, segment in enumerate(segments):
            output_filename = f"{base_name}_segment_{i+1:02d}_{segment['start_time']:.0f}s-{segment['end_time']:.0f}s.mp3"
            output_path = self.output_dir / output_filename
            
            sf.write(output_path, segment['audio'], sr)
            
            print(f"  保存片段 {i+1}: {output_filename} (时长: {segment['duration']:.1f}s)")
        
        # 保存变化点分析图
        self.plot_change_analysis(features, change_points, change_scores, base_name, window_size, sr)
        
        print(f"完成处理: {filename}")
        print("-" * 50)
    
    def plot_change_analysis(self, features, change_points, change_scores, base_name, window_size, sr):
        """
        绘制变化点分析图
        """
        fig, axes = plt.subplots(3, 1, figsize=(30, 20))
        
        # 计算时间轴（秒）
        # 使用extract_audio_features中的默认值
        hop_length = 512
        
        # 特征的时间轴
        feature_times = librosa.frames_to_time(np.arange(len(features)), sr=sr, hop_length=hop_length)
        
        # change_scores的时间轴（从window_size开始）
        change_score_times = librosa.frames_to_time(np.arange(window_size, window_size + len(change_scores)), sr=sr, hop_length=hop_length)
        
        # change_points的时间（已经是帧索引）
        change_point_times = librosa.frames_to_time(change_points, sr=sr, hop_length=hop_length)
        
        # 设置x轴刻度间隔为5秒
        max_time = max(feature_times[-1], change_score_times[-1])
        x_ticks = np.arange(0, max_time + 5, 5)
        
        # 绘制部分特征
        axes[0].plot(feature_times, features[:, 0], label='Energy', alpha=0.7)
        axes[0].plot(feature_times, features[:, 1], label='Spectral Centroid', alpha=0.7)
        axes[0].plot(feature_times, features[:, 2], label='Spectral Bandwidth', alpha=0.7)
        axes[0].set_title('Audio Features')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_xticks(x_ticks)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制MFCC
        axes[1].plot(feature_times, features[:, 5:10], alpha=0.7)
        axes[1].set_title('MFCC Features')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_xticks(x_ticks)
        axes[1].grid(True, alpha=0.3)
        
        # 绘制变化分数和变化点
        axes[2].plot(change_score_times, change_scores, label='Change Score', alpha=0.7)
        for cp_time in change_point_times:
            axes[2].axvline(x=cp_time, color='red', linestyle='--', alpha=0.7)
        axes[2].set_title('Change Point Detection')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_xticks(x_ticks)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{base_name}_change_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_all_files(self):
        """
        处理所有配置的文件
        """
        print("开始处理无人机录音文件...")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 50)
        
        for filename in self.time_ranges.keys():
            self.process_file(filename)
        
        print("所有文件处理完成!")
        print(f"结果保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='无人机录音噪音提取和分割')
    parser.add_argument('--input', default='samples/mp3',
                        help='输入目录路径 (默认: samples/mp3)')
    parser.add_argument('--output', default='samples/extract_noise',
                        help='输出目录路径 (默认: samples/extract_noise)')
    parser.add_argument('--file', 
                        help='处理单个文件（如: 1.mp3）')
    
    args = parser.parse_args()
    
    extractor = DroneNoiseExtractor(args.input, args.output)
    
    if args.file:
        extractor.process_file(args.file)
    else:
        extractor.process_all_files()

if __name__ == "__main__":
    main() 