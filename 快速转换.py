#!/usr/bin/env python3
"""
快速转换脚本：直接转换samples目录中的.m4a文件为MP3格式（更小的文件大小）
"""

import os
import glob
from pydub import AudioSegment

def quick_convert():
    """快速转换samples目录中的.m4a文件为MP3格式"""
    
    # 检查samples目录是否存在
    if not os.path.exists('samples'):
        print("错误: 'samples' 目录不存在")
        return
    
    # 查找所有.m4a文件
    m4a_files = glob.glob("samples/*.wav")
    
    if not m4a_files:
        print("在samples目录中没有找到.m4a文件")
        return
    
    print(f"找到 {len(m4a_files)} 个.m4a文件")
    print("转换格式: MP3 (比特率: 128kbps)")
    print("开始转换...")
    print("-" * 50)
    
    total_original_size = 0
    total_new_size = 0
    
    # 转换每个文件
    for i, m4a_file in enumerate(m4a_files, 1):
        try:
            # 获取文件名（不包含扩展名）
            filename = os.path.splitext(os.path.basename(m4a_file))[0]
            output_file = os.path.join('samples', f"{filename}.mp3")
            
            print(f"({i}/{len(m4a_files)}) 正在转换: {os.path.basename(m4a_file)}")
            
            # 加载音频文件
            audio = AudioSegment.from_file(m4a_file, format="m4a")
            
            # 导出为MP3格式，设置比特率为128k
            audio.export(output_file, format="mp3", bitrate="128k")
            
            # 计算文件大小
            original_size = os.path.getsize(m4a_file) / (1024 * 1024)  # MB
            new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            compression_ratio = (1 - new_size / original_size) * 100
            
            total_original_size += original_size
            total_new_size += new_size
            
            print(f"✓ 转换完成: {os.path.basename(output_file)}")
            print(f"  文件大小: {original_size:.1f}MB → {new_size:.1f}MB (压缩 {compression_ratio:.1f}%)")
            
        except Exception as e:
            print(f"✗ 转换失败 {os.path.basename(m4a_file)}: {str(e)}")
    
    print("-" * 50)
    
    # 显示总体统计
    if total_original_size > 0:
        total_compression = (1 - total_new_size / total_original_size) * 100
        print(f"转换完成!")
        print(f"总体文件大小: {total_original_size:.1f}MB → {total_new_size:.1f}MB")
        print(f"总体压缩率: {total_compression:.1f}%")
        print(f"节省空间: {total_original_size - total_new_size:.1f}MB")

if __name__ == "__main__":
    quick_convert() 