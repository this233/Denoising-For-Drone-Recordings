#!/usr/bin/env python3
"""
无人机录音智能降噪系统 - 主处理模块

本模块实现了一个基于多阶段信号处理技术的无人机录音降噪系统。
系统采用分层降噪架构，结合传统信号处理方法和机器学习算法，
能够有效去除无人机旋翼噪声、电机噪声等各类环境干扰。

主要技术特点：
- 多阶段降噪流水线：频域预处理 → 统计降噪 → 自适应精细降噪 → 残留噪声清理
- 动态噪声匹配：基于多维特征相似度的智能噪声样本选择
- 自适应滤波：结合频谱减法和维纳滤波的混合降噪算法
- 长音频处理：支持任意长度音频的分片处理和无缝重构

作者：[项目作者]
版本：v1.0.0
创建日期：2025年
最后修改：2025年7月

使用示例：
    # 基础用法
    denoiser = DroneVoiceDenoiser()
    denoiser.load_noise_samples()
    denoiser.process_file("input.mp3", "output.mp3")
    
    # 高级用法
    denoiser = DroneVoiceDenoiser(
        noise_dir="custom_noise",
        top_k_noise=8,
        similarity_threshold=0.8
    )
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

# 配置matplotlib中文字体支持
# 解决中文字符在图表中的显示问题
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DroneVoiceDenoiser:
    """
    无人机录音智能降噪器
    
    本类实现了一个完整的无人机录音降噪解决方案，采用多阶段处理架构
    和基于机器学习的噪声匹配算法，能够有效去除各类无人机噪声干扰。
    
    核心算法原理：
    1. 频域预处理：使用带通滤波器保留人声频率范围（50-3400Hz）
    2. 统计降噪：基于噪声统计特性的初步降噪
    3. 自适应精细降噪：
       - 动态噪声匹配：使用多维特征向量计算相似度
       - 频谱减法：S_clean = max(|S_audio|² - α|S_noise|², β|S_audio|²)
       - 维纳滤波：H(f) = S_signal(f) / (S_signal(f) + S_noise(f))
    4. 残留噪声后处理：使用专门的残留噪声样本进行最终清理
    
    特征向量构成：
    - 归一化频谱幅度：反映频域能量分布
    - 归一化功率谱密度：反映频域功率特性
    - 归一化频谱质心：反映频谱能量中心位置
    - 归一化频谱带宽：反映频谱能量分散程度
    - 归一化MFCC系数：反映音频的倒谱特征
    
    相似度计算公式：
    similarity = 0.4 × cosine_similarity + 0.3 × euclidean_similarity + 0.3 × correlation
    
    Attributes:
        noise_dir (Path): 噪声样本库目录路径
        sample_rate (int): 统一处理采样率，默认22050Hz
        chunk_duration (int): 长音频分片长度（秒），默认30秒
        overlap_duration (int): 分片重叠长度（秒），默认5秒
        top_k_noise (int): 动态选择的最佳匹配噪声样本数量
        similarity_threshold (float): 噪声匹配相似度阈值
        noise_samples (list): 存储所有噪声样本的音频数据
        noise_spectra (list): 存储所有噪声样本的频谱特征
        noise_metadata (list): 存储噪声样本的元数据信息
        residual_noise_sample (np.ndarray): 残留噪声样本用于后处理
    
    Notes:
        - 所有音频处理都会重采样到统一采样率以确保一致性
        - 噪声样本库应包含不同工况下的无人机噪声录音
        - 系统支持MP3、WAV、FLAC等主流音频格式
        - 处理过程中会生成详细的频谱分析图表
    """
    
    def __init__(self, noise_dir="noise_samples", top_k_noise=5, similarity_threshold=0.75):
        """
        初始化无人机录音降噪器
        
        设置降噪器的核心参数和配置，建立噪声样本库连接。
        
        Args:
            noise_dir (str): 噪声样本目录路径。应包含不同无人机型号和工况的噪声录音。
                           文件命名格式建议：{drone_id}_segment_{seq}_{start}s-{end}s.mp3
            top_k_noise (int): 动态选择的最佳匹配噪声样本数量。
                              值越大降噪效果可能越好，但计算复杂度增加。
                              推荐范围：3-10
            similarity_threshold (float): 噪声匹配相似度阈值。
                                        只有相似度超过此阈值的样本才参与降噪。
                                        推荐范围：0.6-0.9
        
        Raises:
            FileNotFoundError: 当指定的噪声样本目录不存在时
        
        Note:
            初始化后需要调用 load_noise_samples() 方法加载噪声样本库
        """
        self.noise_dir = Path(noise_dir)
        
        # === 音频处理核心参数 ===
        self.sample_rate = 22050  # 统一采样率，平衡音质和处理效率
        self.chunk_duration = 30  # 分片长度（秒），影响内存使用和处理精度
        self.overlap_duration = 5  # 重叠长度（秒），确保分片边界无缝衔接
        
        # === 噪声匹配参数 ===
        self.top_k_noise = top_k_noise  # Top-K噪声样本选择策略
        self.similarity_threshold = similarity_threshold  # 相似度过滤阈值
        
        # === 噪声样本库存储结构 ===
        self.noise_samples = []  # 原始音频数据列表：List[np.ndarray]
        self.noise_spectra = []  # 频谱特征数据列表：List[np.ndarray] 
        self.noise_metadata = []  # 元数据信息列表：List[Dict]
        self.residual_noise_sample = None  # 残留噪声样本：np.ndarray
        
        print(f"✓ 降噪器初始化完成")
        print(f"  噪声样本目录: {self.noise_dir}")
        print(f"  Top-K样本数: {self.top_k_noise}")
        print(f"  相似度阈值: {self.similarity_threshold:.3f}")
    
    def load_noise_samples(self):
        """
        加载并预处理噪声样本库
        
        从指定目录中加载所有噪声样本文件，提取多维音频特征，
        构建用于动态匹配的噪声样本库。每个样本都会提取以下特征：
        - 频谱幅度分布：用于频谱减法
        - 功率谱密度：用于能量匹配
        - 频谱质心：反映频率中心
        - 频谱带宽：反映频率分散度
        - MFCC系数：反映音色特征
        
        文件命名约定：
        - 格式：{drone_id}_segment_{seq}_{start}s-{end}s.mp3
        - drone_id: 无人机编号（1-10为普通样本，11为残留噪声样本）
        - seq: 片段序号
        - start/end: 时间范围标识
        
        Raises:
            FileNotFoundError: 当噪声样本目录为空或不存在mp3文件时
            ValueError: 当无法建立有效的噪声模型时
        
        Note:
            - 所有样本会被重采样到统一采样率
            - 11号无人机样本被特殊处理作为残留噪声样本
            - 特征向量会被归一化以提高匹配精度
        """
        print("📂 正在加载噪声样本库...")
        
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
        智能噪声匹配算法 - 为音频片段寻找最佳匹配的噪声样本
        
        使用多维特征相似度评估算法，从噪声样本库中选择与当前音频片段
        最相似的Top-K个样本，并进行加权平均以构建自适应噪声模型。
        
        相似度评估算法：
        1. 特征提取：提取音频片段的多维特征向量
        2. 相似度计算：
           - 余弦相似度：cosine_sim = 1 - cosine_distance(feat1, feat2)
           - 欧氏距离相似度：euclidean_sim = 1 / (1 + euclidean_distance)
           - 皮尔逊相关系数：correlation = corrcoef(feat1, feat2)
        3. 综合评分：similarity = 0.4×cosine + 0.3×euclidean + 0.3×correlation
        4. Top-K选择：选择得分最高且超过阈值的前K个样本
        5. 频谱平均：对选中样本的频谱进行加权平均
        
        Args:
            audio_segment (np.ndarray): 待处理的音频片段，形状为(n_samples,)
            top_k (int): 选择的最佳匹配样本数量，推荐范围[3,10]
            similarity_threshold (float): 相似度阈值过滤器，推荐范围[0.6,0.9]
            
        Returns:
            tuple: (averaged_noise_spectrum, best_matches_info)
                - averaged_noise_spectrum (np.ndarray | None): 
                  平均后的噪声频谱，如果没有符合条件的样本则返回None
                - best_matches_info (list): 最佳匹配样本的详细信息列表
                  每个元素包含：similarity, metadata, cosine_sim, euclidean_sim, correlation
        
        Note:
            - 算法会自动处理特征向量长度不一致的问题
            - 当没有样本超过相似度阈值时，会跳过基于噪声模型的降噪
            - 返回的频谱已经过归一化处理，可直接用于频谱减法
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
        多阶段智能降噪核心算法
        
        本方法实现了四阶段渐进式降噪流水线，每个阶段针对不同类型的噪声
        采用专门的处理策略，确保在有效去除噪声的同时保持音频质量。
        
        降噪流水线详解：
        
        【阶段1：频域预处理】
        - 目标：去除明显的频域噪声，保留人声关键频率
        - 方法：双向带通滤波器设计
          * 高通滤波（50Hz）：去除极低频噪声（电机振动、风噪等）
          * 低通滤波（3400Hz）：去除高频干扰（电子噪声、数字噪声等）
        - 保留范围：50-3400Hz（覆盖人声基频和主要谐波）
        
        【阶段2：统计降噪】
        - 目标：基于噪声统计特性进行初步清理
        - 方法：使用noisereduce库的平稳噪声假设算法
        - 参数：stationary=True, prop_decrease=0.8
        - 特点：保守降噪，避免过度处理导致失真
        
        【阶段3：自适应精细降噪】
        - 目标：基于噪声库的智能匹配进行精准降噪
        - 方法：
          * 动态噪声匹配：寻找最相似的Top-K噪声样本
          * 频谱减法：S_clean = max(|S|² - α|N|², β|S|²)
          * 自适应维纳滤波：H(f) = S_signal/(S_signal + S_noise)
        - 自适应特性：根据相似度动态调整降噪强度
        
        【阶段4：残留噪声后处理】
        - 目标：清理前三阶段未能去除的残留噪声
        - 方法：使用专门的残留噪声样本进行最终清理
        - 参数：stationary=False, prop_decrease=0.6
        - 特点：非平稳噪声假设，适应动态噪声环境
        
        Args:
            y (np.ndarray): 输入音频信号，形状为(n_samples,)
            sr (int): 音频采样率，建议与self.sample_rate一致
            return_stages (bool): 是否返回各阶段的中间结果，用于分析和调试
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
                - 当return_stages=False时：返回最终降噪结果
                - 当return_stages=True时：返回(最终结果, 各阶段结果列表)
                  阶段结果列表包含：[原始, 阶段1, 阶段2, 阶段3, 阶段4]
        
        Raises:
            ValueError: 当输入音频为空或采样率不匹配时
            
        Note:
            - 算法会根据噪声样本库的匹配情况自动调整处理策略
            - 当找不到合适的噪声样本时，会跳过阶段3的精细降噪
            - 每个阶段都保持音频的时域连续性，避免产生人工痕迹
            - 建议在调用前确保已加载噪声样本库
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
        
        # 计算图片保存路径（在输出文件同目录下）
        output_path = Path(output_file)
        plot_save_path = output_path.parent / output_path.stem
        
        # # 生成详细对比图 - 只对第一个片段进行分阶段分析
        # if num_chunks > 0:
        #     # 对第一个片段进行详细分析
        #     first_chunk = y[:chunk_samples] if len(y) >= chunk_samples else y
        #     denoised_chunk, stage_results = self.multi_stage_denoise(
        #         first_chunk, self.sample_rate, return_stages=True
        #     )
        #     self.plot_comparison(first_chunk, denoised_chunk, f"{plot_save_path}_stages", stage_results)
        
        # 生成整体对比图
        self.plot_comparison(y, final_audio, str(plot_save_path))
        
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
    
    def plot_comparison(self, original, denoised, plot_save_path, stage_results=None):
        """
        绘制降噪前后的对比图，可选显示各阶段频谱
        
        Args:
            original: 原始音频
            denoised: 降噪后音频
            plot_save_path: 图片保存的完整路径（不含扩展名）
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
        plt.savefig(f"{plot_save_path}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"对比图保存至: {plot_save_path}_comparison.png")
    
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
    
    def process_file(self, input_file, output_file):
        """
        处理单个音频文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        output_path = Path(output_file)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理音频
        result = self.process_long_audio(input_path, output_path)
        
        return result
    
    def batch_process(self, input_dir, output_dir, pattern="*.mp3"):
        """
        批量处理音频文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
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
                # 构建输出文件路径
                output_file = output_path / f"{file.stem}_denoised{file.suffix}"
                self.process_file(file, output_file)
            except Exception as e:
                print(f"处理文件 {file.name} 时出错: {e}")
                continue
        
        print(f"\n批量处理完成！结果保存在: {output_path}")

def main():
    """
    无人机录音降噪系统命令行接口主程序
    
    提供完整的命令行参数解析、输入验证、降噪器初始化和音频处理功能。
    支持单文件和批量处理两种模式，具有完善的错误处理和用户反馈机制。
    
    命令行参数：
        --input: 输入音频文件或目录路径（必需）
        --output: 输出文件或目录路径（必需）
        --noise-dir: 噪声样本库目录路径（默认：noise_samples）
        --batch: 启用批量处理模式标志
        --pattern: 批量处理时的文件匹配模式（默认：*.mp3）
        --top-k: 噪声匹配Top-K数量（默认：5）
        --similarity-threshold: 相似度阈值（默认：0.75）
    
    处理流程：
        1. 解析命令行参数
        2. 验证输入输出路径的有效性
        3. 初始化降噪器实例
        4. 加载噪声样本库
        5. 执行音频降噪处理
        6. 输出处理结果和状态信息
    
    错误处理：
        - 路径验证：确保输入输出路径符合处理模式要求
        - 异常捕获：处理文件加载、降噪算法和文件保存异常
        - 用户友好：提供清晰的错误信息和解决建议
    
    Examples:
        # 单文件处理
        python drone_denoiser.py --input recording.mp3 --output clean.mp3
        
        # 批量处理
        python drone_denoiser.py --batch --input ./recordings --output ./cleaned
        
        # 自定义参数
        python drone_denoiser.py --input recording.mp3 --output clean.mp3 \\
                                --top-k 8 --similarity-threshold 0.8
    """
    parser = argparse.ArgumentParser(description='无人机录音智能降噪系统 - 专业级音频降噪解决方案')
    parser.add_argument('--input', required=True, help='输入音频文件或目录')
    parser.add_argument('--output', required=True, help='输出文件或目录')
    parser.add_argument('--noise-dir', default='noise_samples/segments', help='噪声样本目录')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--pattern', default='*.mp3', help='批量处理时的文件匹配模式')
    parser.add_argument('--top-k', type=int, default=5, help='选择前k个最匹配的噪声样本进行平均（默认：5）')
    parser.add_argument('--similarity-threshold', type=float, default=0.75, help='相似度阈值，只有超过此值的样本才会被选择（默认：0.75）')
    
    args = parser.parse_args()
    
    # 验证输入输出参数
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.batch:
        # 批量模式：输入必须是目录，输出必须是目录
        if not input_path.is_dir():
            print(f"错误：批量模式下输入必须是目录，但 {args.input} 不是目录")
            return
        # 输出路径可以不存在（会自动创建），但如果存在必须是目录
        if output_path.exists() and not output_path.is_dir():
            print(f"错误：批量模式下输出必须是目录，但 {args.output} 不是目录")
            return
    else:
        # 单文件模式：输入必须是文件，输出必须是文件路径
        if not input_path.is_file():
            print(f"错误：单文件模式下输入必须是文件，但 {args.input} 不是文件")
            return
        # 输出路径如果存在，必须是文件；如果不存在，其父目录必须存在或可以创建
        if output_path.exists() and output_path.is_dir():
            print(f"错误：单文件模式下输出必须是文件路径，但 {args.output} 是目录")
            return
    
    # 创建降噪器
    denoiser = DroneVoiceDenoiser(
        noise_dir=args.noise_dir,
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
            denoiser.batch_process(args.input, args.output, args.pattern)
        else:
            denoiser.process_file(args.input, args.output)
    except Exception as e:
        print(f"处理音频失败: {e}")
        return
    
    print("降噪处理完成！")

if __name__ == "__main__":
    main() 