#!/usr/bin/env python3
"""
无人机录音综合降噪脚本
利用多种噪声样本建立综合噪声模型，对录音进行多层次降噪处理
"""

import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt, stft, istft
from scipy.stats import mode
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DroneVoiceDenoiser:
    def __init__(self, noise_dir="samples/noise", output_dir="samples/denoised", top_k_noise=5, similarity_threshold=0.75):
        """
        初始化无人机录音降噪器
        
        Args:
            noise_dir: 噪声样本目录
            output_dir: 输出目录
            top_k_noise: 选择前k个最匹配的噪声样本进行平均
            similarity_threshold: 相似度阈值，只有超过此值的样本才会被选择
        """
        self.noise_dir = Path(noise_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 降噪参数
        self.sample_rate = 22050  # 统一采样率
        self.chunk_duration = 30  # 分片长度（秒）
        self.overlap_duration = 5  # 重叠长度（秒）
        self.top_k_noise = top_k_noise  # 选择前k个最匹配的噪声样本
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        
        # 噪声样本库 - 存储所有单独的噪声样本
        self.noise_samples = []  # 存储所有噪声样本的音频数据
        self.noise_spectra = []  # 存储所有噪声样本的频谱特征
        self.noise_metadata = []  # 存储噪声样本的元数据（文件名、无人机编号等）
        self.residual_noise_sample = None  # 残留噪声样本
        
        # 频率范围（无人机噪声主要集中在低频和中频）
        self.drone_freq_ranges = [
            (20, 200),    # 低频螺旋桨噪声
            (200, 1000),  # 中低频电机噪声
            (1000, 4000), # 中频风切声
            (4000, 8000)  # 高频谐波
        ]
        
        print(f"初始化降噪器完成")
        print(f"噪声样本目录: {self.noise_dir}")
        print(f"输出目录: {self.output_dir}")
    
    def load_noise_samples(self):
        """
        加载所有噪声样本，保存单独的样本而不是平均值
        """
        print("正在加载噪声样本...")
        
        noise_files = list(self.noise_dir.glob("*.mp3"))
        if not noise_files:
            raise FileNotFoundError(f"在 {self.noise_dir} 中未找到噪声样本文件")
        
        # 按无人机编号分组统计
        drone_groups = {}
        valid_samples = 0
        
        for file in noise_files:
            # 提取无人机编号（如 1_segment_01_90s-119s.mp3 -> 1）
            drone_num = file.name.split('_')[0]
            if drone_num not in drone_groups:
                drone_groups[drone_num] = 0
            
            try:
                # 加载音频并重采样
                y, sr = librosa.load(file, sr=self.sample_rate)
                
                # 跳过无人机11的样本（用作残留噪声）
                if drone_num == '11':
                    continue
                
                # 提取频谱特征
                stft_matrix = librosa.stft(y, n_fft=2048, hop_length=512)
                magnitude_spectrum = np.abs(stft_matrix)
                
                # 计算平均频谱作为特征
                avg_spectrum = np.mean(magnitude_spectrum, axis=1)
                
                # 计算功率谱密度
                power_spectrum = np.mean(magnitude_spectrum ** 2, axis=1)
                
                # 计算频谱质心和带宽作为额外特征
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)[0]
                
                # 计算MFCC特征
                mfcc_features = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=13)
                mfcc_mean = np.mean(mfcc_features, axis=1)
                
                # 组合特征向量
                feature_vector = np.concatenate([
                    avg_spectrum / np.max(avg_spectrum),  # 归一化频谱
                    power_spectrum / np.max(power_spectrum),  # 归一化功率谱
                    [np.mean(spectral_centroid) / (self.sample_rate / 2)],  # 归一化质心
                    [np.mean(spectral_bandwidth) / (self.sample_rate / 2)],  # 归一化带宽
                    mfcc_mean / np.max(np.abs(mfcc_mean))  # 归一化MFCC
                ])
                
                # 存储噪声样本
                self.noise_samples.append(y)
                self.noise_spectra.append(avg_spectrum)
                self.noise_metadata.append({
                    'file': file.name,
                    'drone_num': drone_num,
                    'feature_vector': feature_vector,
                    'avg_spectrum': avg_spectrum,
                    'power_spectrum': power_spectrum
                })
                
                drone_groups[drone_num] += 1
                valid_samples += 1
                
            except Exception as e:
                print(f"  警告: 无法加载文件 {file}: {e}")
                continue
        
        print(f"成功加载 {valid_samples} 个噪声样本:")
        for drone_num, count in drone_groups.items():
            if drone_num != '11':
                print(f"  无人机 {drone_num}: {count} 个样本")
        
        if valid_samples == 0:
            raise ValueError("无法建立噪声模型：没有有效的噪声样本")
        
        # 加载残留噪声样本
        self.load_residual_noise_sample()
    
    def find_best_matching_noise(self, audio_segment, top_k=5, similarity_threshold=0.75):
        """
        为音频片段找到最匹配的多个噪声样本并进行平均
        
        Args:
            audio_segment: 音频片段
            top_k: 选择前k个最匹配的样本进行平均
            similarity_threshold: 相似度阈值，只有超过此值的样本才会被选择
            
        Returns:
            averaged_noise_spectrum: 平均后的噪声频谱（如果没有合适样本则返回None）
            best_matches_info: 最匹配噪声样本的信息列表
        """
        if len(self.noise_metadata) == 0:
            raise ValueError("没有可用的噪声样本")
        
        # 限制top_k不超过可用样本数
        top_k = min(top_k, len(self.noise_metadata))
        
        # 计算音频片段的特征
        stft_matrix = librosa.stft(audio_segment, n_fft=2048, hop_length=512)
        magnitude_spectrum = np.abs(stft_matrix)
        avg_spectrum = np.mean(magnitude_spectrum, axis=1)
        power_spectrum = np.mean(magnitude_spectrum ** 2, axis=1)
        
        # 计算额外特征
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate)[0]
        mfcc_features = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc_features, axis=1)
        
        # 组合特征向量
        audio_feature_vector = np.concatenate([
            avg_spectrum / np.max(avg_spectrum),
            power_spectrum / np.max(power_spectrum),
            [np.mean(spectral_centroid) / (self.sample_rate / 2)],
            [np.mean(spectral_bandwidth) / (self.sample_rate / 2)],
            mfcc_mean / np.max(np.abs(mfcc_mean))
        ])
        
        # 计算与所有噪声样本的相似度
        similarities = []
        
        for metadata in self.noise_metadata:
            noise_feature = metadata['feature_vector']
            
            # 确保特征向量长度一致
            min_len = min(len(audio_feature_vector), len(noise_feature))
            audio_feat_trimmed = audio_feature_vector[:min_len]
            noise_feat_trimmed = noise_feature[:min_len]
            
            # 计算多种相似度度量
            # 1. 余弦相似度
            cosine_sim = 1 - cosine(audio_feat_trimmed, noise_feat_trimmed)
            
            # 2. 欧氏距离相似度
            euclidean_dist = np.linalg.norm(audio_feat_trimmed - noise_feat_trimmed)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # 3. 皮尔逊相关系数
            correlation = np.corrcoef(audio_feat_trimmed, noise_feat_trimmed)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # 组合相似度得分
            combined_similarity = (0.4 * cosine_sim + 0.3 * euclidean_sim + 0.3 * abs(correlation))
            
            similarities.append({
                'similarity': combined_similarity,
                'metadata': metadata,
                'cosine_sim': cosine_sim,
                'euclidean_sim': euclidean_sim,
                'correlation': correlation
            })
        
        # 按相似度排序，选择前top_k个
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 过滤掉相似度低于阈值的样本
        qualified_matches = [s for s in similarities if s['similarity'] >= similarity_threshold]
        
        if not qualified_matches:
            print(f"  ⚠️  没有找到相似度≥{similarity_threshold:.3f}的噪声样本，跳过谱减法和维纳滤波")
            print(f"     最高相似度: {similarities[0]['similarity']:.3f} ({similarities[0]['metadata']['file']})")
            return None, []
        
        # 选择前top_k个合格的样本
        best_matches = qualified_matches[:top_k]
        
        # 打印匹配信息
        print(f"  ✓ 找到{len(qualified_matches)}个相似度≥{similarity_threshold:.3f}的噪声样本")
        print(f"  选择前{len(best_matches)}个最匹配的噪声样本进行平均:")
        total_similarity = 0
        selected_spectra = []
        weights = []
        
        for i, match in enumerate(best_matches):
            print(f"    {i+1}. {match['metadata']['file']}")
            print(f"       相似度: {match['similarity']:.3f} (余弦:{match['cosine_sim']:.3f}, 欧氏:{match['euclidean_sim']:.3f}, 相关:{match['correlation']:.3f})")
            
            # 收集频谱和权重
            selected_spectra.append(match['metadata']['avg_spectrum'])
            weights.append(match['similarity'])
            total_similarity += match['similarity']
        
        # 对频谱进行加权平均
        if total_similarity > 0:
            # 归一化权重
            weights = np.array(weights) / total_similarity
            
            # 加权平均
            averaged_spectrum = np.zeros_like(selected_spectra[0])
            for spectrum, weight in zip(selected_spectra, weights):
                averaged_spectrum += spectrum * weight
                
            print(f"    使用加权平均，权重分布: {[f'{w:.3f}' for w in weights]}")
        else:
            # 如果所有相似度都是0，使用简单平均
            averaged_spectrum = np.mean(selected_spectra, axis=0)
            print(f"    使用简单平均")
        
        return averaged_spectrum, best_matches
    
    def load_residual_noise_sample(self):
        """
        加载残留噪声样本用于noisereduce
        """
        residual_noise_file = self.noise_dir / "11_segment_01_14s-15s,17s-25s.mp3"
        
        if residual_noise_file.exists():
            try:
                self.residual_noise_sample, _ = librosa.load(residual_noise_file, sr=self.sample_rate)
                print(f"✓ 加载残留噪声样本: {residual_noise_file.name}")
                print(f"  噪声样本长度: {len(self.residual_noise_sample)/self.sample_rate:.1f}秒")
            except Exception as e:
                print(f"⚠️  加载残留噪声样本失败: {e}")
                self.residual_noise_sample = None
        else:
            print(f"⚠️  残留噪声样本文件不存在: {residual_noise_file}")
            self.residual_noise_sample = None
    
    def spectral_subtraction(self, audio_spectrum, noise_spectrum, alpha=2.0, beta=0.01):
        """
        频谱减除法降噪
        
        Args:
            audio_spectrum: 音频频谱
            noise_spectrum: 噪声频谱
            alpha: 过减除因子
            beta: 频谱下限因子
        """
        # 计算功率谱
        audio_power = np.abs(audio_spectrum) ** 2
        noise_power = np.abs(noise_spectrum) ** 2
        
        # 频谱减除
        clean_power = audio_power - alpha * noise_power
        
        # 设置频谱下限
        clean_power = np.maximum(clean_power, beta * audio_power)
        
        # 保持相位不变
        clean_spectrum = np.sqrt(clean_power) * np.exp(1j * np.angle(audio_spectrum))
        
        return clean_spectrum
    
    def adaptive_wiener_filter(self, audio_spectrum, noise_spectrum, frame_idx):
        """
        自适应维纳滤波
        
        Args:
            audio_spectrum: 音频频谱
            noise_spectrum: 噪声频谱
            frame_idx: 帧索引（用于自适应）
        """
        # 估计信号功率
        audio_power = np.abs(audio_spectrum) ** 2
        noise_power = np.abs(noise_spectrum) ** 2
        
        # 自适应调整噪声估计
        adaptive_factor = 1.0 + 0.3 * np.sin(frame_idx * 0.01)  # 简单的时变因子
        adjusted_noise_power = noise_power * adaptive_factor
        
        # 维纳滤波
        signal_power = np.maximum(audio_power - adjusted_noise_power, 
                                 0.1 * audio_power)
        
        wiener_gain = signal_power / (signal_power + adjusted_noise_power)
        
        return audio_spectrum * wiener_gain
    
    def frequency_domain_filter(self, y, sr):
        """
        频域滤波，针对无人机噪声的特定频率范围
        """
        # 应用带通滤波，保留人声频率范围
        # 人声主要集中在 85-255Hz (基频) 和 255-2000Hz (泛音)
        
        # 设计高通滤波器，去除极低频噪声
        b, a = butter(4, 50, btype='high', fs=sr)
        y_filtered = filtfilt(b, a, y)
        
        # 设计低通滤波器，去除高频噪声
        b, a = butter(4, 3400, btype='low', fs=sr)
        y_filtered = filtfilt(b, a, y_filtered)
        
        return y_filtered
    
    def multi_stage_denoise(self, y, sr, return_stages=False):
        """
        多阶段降噪处理，动态选择最佳噪声样本
        
        Args:
            y: 输入音频信号
            sr: 采样率
            return_stages: 是否返回各阶段结果
            
        Returns:
            如果return_stages=True: (final_audio, stage_results)
            否则: final_audio
        """
        # 保存各阶段结果
        if return_stages:
            stage_results = [y.copy()]  # 原始音频
        
        # 第一阶段：频域预处理
        y_stage1 = self.frequency_domain_filter(y, sr)
        if return_stages:
            stage_results.append(y_stage1.copy())
        
        # 第二阶段：基于noisereduce的初步降噪
        y_stage2 = nr.reduce_noise(y=y_stage1, sr=sr, stationary=True, prop_decrease=0.8)
        if return_stages:
            stage_results.append(y_stage2.copy())
        
        # 第三阶段：基于动态选择的噪声样本进行精细降噪
        print("  选择最佳匹配噪声样本...")
        averaged_noise_spectrum, best_matches = self.find_best_matching_noise(y_stage2, top_k=self.top_k_noise, similarity_threshold=self.similarity_threshold)
        
        if averaged_noise_spectrum is not None:
            # 有合适的噪声样本，进行频谱减除和维纳滤波
            print("  进行基于噪声模型的精细降噪...")
            
            # 计算STFT
            stft_matrix = librosa.stft(y_stage2, n_fft=2048, hop_length=512)
            
            # 对每一帧进行降噪
            denoised_stft = np.zeros_like(stft_matrix)
            for i in range(stft_matrix.shape[1]):
                frame_spectrum = stft_matrix[:, i]
                
                # 使用频谱减除
                denoised_frame = self.spectral_subtraction(
                    frame_spectrum, 
                    averaged_noise_spectrum[:len(frame_spectrum)],
                    alpha=1.5,
                    beta=0.05
                )
                
                # 使用自适应维纳滤波
                denoised_frame = self.adaptive_wiener_filter(
                    denoised_frame,
                    averaged_noise_spectrum[:len(frame_spectrum)],
                    i
                )
                
                denoised_stft[:, i] = denoised_frame
            
            # 重构音频
            y_stage3 = librosa.istft(denoised_stft, hop_length=512, length=len(y_stage2))
        else:
            # 没有合适的噪声样本，跳过频谱减除和维纳滤波
            print("  跳过基于噪声模型的精细降噪")
            y_stage3 = y_stage2.copy()
        if return_stages:
            stage_results.append(y_stage3.copy())
        
        # 第四阶段：后处理 - 使用残留噪声样本
        if self.residual_noise_sample is not None:
            # 再次动态选择噪声样本进行最终清理
            print("  使用残留噪声样本进行最终降噪...")
            
            # 找到11号无人机的噪声样本
            residual_noise_metadata = None
            for metadata in self.noise_metadata:
                if metadata['drone_num'] == '11':
                    residual_noise_metadata = metadata
                    break
            
            if residual_noise_metadata is None:
                # 如果没有11号样本在列表中，使用直接加载的残留噪声
                y_final = nr.reduce_noise(
                    y=y_stage3, 
                    sr=sr, 
                    y_noise=self.residual_noise_sample,
                    stationary=False, 
                    prop_decrease=0.6
                )
            else:
                # 使用残留噪声频谱进行最后一轮谱减法
                residual_stft = librosa.stft(y_stage3, n_fft=2048, hop_length=512)
                final_denoised_stft = np.zeros_like(residual_stft)
                
                for i in range(residual_stft.shape[1]):
                    frame_spectrum = residual_stft[:, i]
                    
                    # 最终频谱减除
                    denoised_frame = self.spectral_subtraction(
                        frame_spectrum, 
                        residual_noise_metadata['avg_spectrum'][:len(frame_spectrum)],
                        alpha=2.0,
                        beta=0.01
                    )
                    
                    final_denoised_stft[:, i] = denoised_frame
                
                y_final = librosa.istft(final_denoised_stft, hop_length=512, length=len(y_stage3))
        else:
            # 使用默认方法
            y_final = nr.reduce_noise(y=y_stage3, sr=sr, stationary=True, prop_decrease=0.6)
            print("  使用默认方法进行最终降噪")
        
        if return_stages:
            stage_results.append(y_final.copy())
        
        if return_stages:
            return y_final, stage_results
        else:
            return y_final
    
    def process_long_audio(self, input_file, output_file):
        """
        处理长音频文件（分片处理）
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        print(f"开始处理音频文件: {input_file}")
        
        # 加载音频
        y, original_sr = librosa.load(input_file, sr=None)
        
        # 重采样到统一采样率
        if original_sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=self.sample_rate)
        
        duration = len(y) / self.sample_rate
        print(f"音频总时长: {duration:.1f}秒")
        
        # 计算分片参数
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        # 分片处理
        processed_chunks = []
        num_chunks = int(np.ceil(len(y) / step_samples))
        
        print(f"将分为 {num_chunks} 个片段处理...")
        
        for i in range(num_chunks):
            start_idx = i * step_samples
            end_idx = min(start_idx + chunk_samples, len(y))
            
            chunk = y[start_idx:end_idx]
            
            print(f"处理片段 {i+1}/{num_chunks} ({start_idx/self.sample_rate:.1f}s - {end_idx/self.sample_rate:.1f}s)")
            
            # 降噪处理
            denoised_chunk = self.multi_stage_denoise(chunk, self.sample_rate)
            
            processed_chunks.append((start_idx, end_idx, denoised_chunk))
        
        # 重组音频
        print("重组处理后的音频...")
        final_audio = self.reconstruct_audio(processed_chunks, len(y), overlap_samples)
        
        # 保存结果
        sf.write(output_file, final_audio, self.sample_rate)
        print(f"降噪完成，保存至: {output_file}")
        
        # 生成详细对比图 - 只对第一个片段进行分阶段分析
        if num_chunks > 0:
            # 对第一个片段进行详细分析
            first_chunk = y[:chunk_samples] if len(y) >= chunk_samples else y
            denoised_chunk, stage_results = self.multi_stage_denoise(
                first_chunk, self.sample_rate, return_stages=True
            )
            self.plot_comparison(first_chunk, denoised_chunk, f"{input_file.stem}_stages", stage_results)
        
        # 生成整体对比图
        self.plot_comparison(y, final_audio, input_file.stem)
        
        return final_audio
    
    def reconstruct_audio(self, processed_chunks, total_length, overlap_samples):
        """
        重组分片处理后的音频
        """
        final_audio = np.zeros(total_length)
        weight_sum = np.zeros(total_length)
        
        for start_idx, end_idx, chunk in processed_chunks:
            chunk_length = len(chunk)
            
            # 创建权重窗口（汉宁窗）
            window = np.ones(chunk_length)
            if chunk_length > overlap_samples:
                # 开始部分的渐变
                fade_in = np.linspace(0, 1, overlap_samples // 2)
                window[:len(fade_in)] = fade_in
                
                # 结束部分的渐变
                fade_out = np.linspace(1, 0, overlap_samples // 2)
                window[-len(fade_out):] = fade_out
            
            # 累加音频和权重
            end_actual = min(start_idx + chunk_length, total_length)
            chunk_actual = chunk[:end_actual - start_idx]
            window_actual = window[:end_actual - start_idx]
            
            final_audio[start_idx:end_actual] += chunk_actual * window_actual
            weight_sum[start_idx:end_actual] += window_actual
        
        # 归一化
        weight_sum[weight_sum == 0] = 1  # 避免除零
        final_audio = final_audio / weight_sum
        
        return final_audio
    
    def plot_comparison(self, original, denoised, filename, stage_results=None):
        """
        绘制降噪前后的对比图，可选显示各阶段频谱
        
        Args:
            original: 原始音频
            denoised: 降噪后音频
            filename: 文件名
            stage_results: 各阶段结果列表 [原始, 阶段1, 阶段2, 阶段3, 最终]
        """
        if stage_results is not None:
            # 显示各阶段频谱的详细对比图
            fig, axes = plt.subplots(3, 2, figsize=(20, 15))
            
            # 阶段名称
            stage_names = [
                '原始音频',
                '阶段1: 频域滤波',
                '阶段2: 初步降噪',
                '阶段3: 精细降噪',
                '阶段4: 后处理'
            ]
            
            # 颜色
            colors = ['blue', 'green', 'orange', 'purple', 'red']
            
            # 第一行：时域波形对比
            time_orig = np.arange(len(stage_results[0])) / self.sample_rate
            for i, (audio, name, color) in enumerate(zip(stage_results, stage_names, colors)):
                if i < 3:  # 前3个阶段
                    time = np.arange(len(audio)) / self.sample_rate
                    axes[0, 0].plot(time, audio, alpha=0.6, label=name, color=color)
            
            axes[0, 0].set_title('时域波形对比 (前3阶段)')
            axes[0, 0].set_xlabel('时间 (秒)')
            axes[0, 0].set_ylabel('幅度')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 后2个阶段
            for i, (audio, name, color) in enumerate(zip(stage_results[3:], stage_names[3:], colors[3:])):
                time = np.arange(len(audio)) / self.sample_rate
                axes[0, 1].plot(time, audio, alpha=0.6, label=name, color=color)
            
            axes[0, 1].set_title('时域波形对比 (后2阶段)')
            axes[0, 1].set_xlabel('时间 (秒)')
            axes[0, 1].set_ylabel('幅度')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 第二行：频谱对比 (前3阶段)
            for i, (audio, name, color) in enumerate(zip(stage_results[:3], stage_names[:3], colors[:3])):
                freqs, spectrum = self.compute_spectrum(audio)
                axes[1, 0].plot(freqs, spectrum, alpha=0.7, label=name, color=color)
            
            axes[1, 0].set_title('频谱对比 (前3阶段)')
            axes[1, 0].set_xlabel('频率 (Hz)')
            axes[1, 0].set_ylabel('幅度 (dB)')
            axes[1, 0].set_xlim(0, 8000)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 频谱对比 (后2阶段)
            for i, (audio, name, color) in enumerate(zip(stage_results[3:], stage_names[3:], colors[3:])):
                freqs, spectrum = self.compute_spectrum(audio)
                axes[1, 1].plot(freqs, spectrum, alpha=0.7, label=name, color=color)
            
            axes[1, 1].set_title('频谱对比 (后2阶段)')
            axes[1, 1].set_xlabel('频率 (Hz)')
            axes[1, 1].set_ylabel('幅度 (dB)')
            axes[1, 1].set_xlim(0, 8000)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 第三行：所有阶段频谱叠加图
            for i, (audio, name, color) in enumerate(zip(stage_results, stage_names, colors)):
                freqs, spectrum = self.compute_spectrum(audio)
                axes[2, 0].plot(freqs, spectrum, alpha=0.6, label=name, color=color)
            
            axes[2, 0].set_title('所有阶段频谱叠加对比')
            axes[2, 0].set_xlabel('频率 (Hz)')
            axes[2, 0].set_ylabel('幅度 (dB)')
            axes[2, 0].set_xlim(0, 8000)
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 频谱差异分析
            freqs_orig, spectrum_orig = self.compute_spectrum(stage_results[0])
            freqs_final, spectrum_final = self.compute_spectrum(stage_results[-1])
            spectrum_diff = spectrum_orig - spectrum_final
            
            axes[2, 1].plot(freqs_orig, spectrum_diff, color='black', alpha=0.7, label='降噪量 (原始-最终)')
            axes[2, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[2, 1].set_title('降噪量分析 (dB)')
            axes[2, 1].set_xlabel('频率 (Hz)')
            axes[2, 1].set_ylabel('降噪量 (dB)')
            axes[2, 1].set_xlim(0, 8000)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
        else:
            # 简单的前后对比图
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # 时域对比
            time_orig = np.arange(len(original)) / self.sample_rate
            time_denoised = np.arange(len(denoised)) / self.sample_rate
            
            axes[0].plot(time_orig, original, alpha=0.7, label='原始音频')
            axes[0].set_title('原始音频波形')
            axes[0].set_xlabel('时间 (秒)')
            axes[0].set_ylabel('幅度')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_denoised, denoised, alpha=0.7, label='降噪后', color='red')
            axes[1].set_title('降噪后音频波形')
            axes[1].set_xlabel('时间 (秒)')
            axes[1].set_ylabel('幅度')
            axes[1].grid(True, alpha=0.3)
            
            # 频谱对比
            freqs_orig, fft_orig = self.compute_spectrum(original)
            freqs_denoised, fft_denoised = self.compute_spectrum(denoised)
            
            axes[2].plot(freqs_orig, fft_orig, alpha=0.7, label='原始音频')
            axes[2].set_title('原始音频频谱')
            axes[2].set_xlabel('频率 (Hz)')
            axes[2].set_ylabel('幅度 (dB)')
            axes[2].set_xlim(0, 8000)
            axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(freqs_denoised, fft_denoised, alpha=0.7, label='降噪后', color='red')
            axes[3].set_title('降噪后音频频谱')
            axes[3].set_xlabel('频率 (Hz)')
            axes[3].set_ylabel('幅度 (dB)')
            axes[3].set_xlim(0, 8000)
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"对比图保存至: {self.output_dir / f'{filename}_comparison.png'}")
    
    def compute_spectrum(self, audio):
        """
        计算音频频谱
        """
        # 计算FFT
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # 转换为dB
        positive_fft_db = 20 * np.log10(positive_fft + 1e-10)
        
        return positive_freqs, positive_fft_db
    
    def process_file(self, input_file, output_file=None):
        """
        处理单个音频文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        if output_file is None:
            output_file = self.output_dir / f"{input_path.stem}_denoised{input_path.suffix}"
        else:
            output_file = Path(output_file)
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理音频
        result = self.process_long_audio(input_path, output_file)
        
        return result
    
    def batch_process(self, input_dir, pattern="*.mp3"):
        """
        批量处理音频文件
        
        Args:
            input_dir: 输入目录
            pattern: 文件匹配模式
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        # 查找匹配的文件
        audio_files = list(input_path.glob(pattern))
        
        if not audio_files:
            print(f"在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return
        
        print(f"找到 {len(audio_files)} 个文件待处理")
        
        # 批量处理
        for i, file in enumerate(audio_files):
            print(f"\n处理文件 {i+1}/{len(audio_files)}: {file.name}")
            try:
                self.process_file(file)
            except Exception as e:
                print(f"处理文件 {file.name} 时出错: {e}")
                continue
        
        print(f"\n批量处理完成！结果保存在: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='无人机录音综合降噪处理')
    parser.add_argument('--input', required=True, help='输入音频文件或目录')
    parser.add_argument('--output', help='输出文件或目录（可选）')
    parser.add_argument('--noise-dir', default='samples/noise', help='噪声样本目录')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--pattern', default='*.mp3', help='批量处理时的文件匹配模式')
    parser.add_argument('--top-k', type=int, default=5, help='选择前k个最匹配的噪声样本进行平均（默认：5）')
    parser.add_argument('--similarity-threshold', type=float, default=0.75, help='相似度阈值，只有超过此值的样本才会被选择（默认：0.75）')
    
    args = parser.parse_args()
    
    # 创建降噪器
    denoiser = DroneVoiceDenoiser(
        noise_dir=args.noise_dir,
        output_dir=args.output or 'samples/denoised',
        top_k_noise=args.top_k,
        similarity_threshold=args.similarity_threshold
    )
    
    # 加载噪声样本
    try:
        denoiser.load_noise_samples()
    except Exception as e:
        print(f"加载噪声样本失败: {e}")
        print("请确保噪声样本目录存在且包含mp3文件")
        return
    
    # 处理音频
    try:
        if args.batch:
            denoiser.batch_process(args.input, args.pattern)
        else:
            denoiser.process_file(args.input, args.output)
    except Exception as e:
        print(f"处理音频失败: {e}")
        return
    
    print("降噪处理完成！")

if __name__ == "__main__":
    main() 