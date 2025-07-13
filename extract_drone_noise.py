#!/usr/bin/env python3
"""
无人机录音噪音提取和分割脚本
从指定时间段提取噪音，并根据音频特征变化进行智能分割
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

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DroneNoiseExtractor:
    def __init__(self, input_dir="samples/mp3", output_dir="samples/extract_noise"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件时间配置
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
        
        self.max_segment_length = 30  # 最大片段长度（秒）
        self.min_segment_length = 10   # 最小片段长度（秒）
        
    def extract_audio_features(self, y, sr, hop_length=512):
        """
        提取音频特征用于变化点检测
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
        检测音频特征变化点
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
        处理单个音频文件
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