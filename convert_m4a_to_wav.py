#!/usr/bin/env python3
"""
音频格式转换脚本：将.m4a文件转换为多种音频格式
"""

import os
import glob
from pydub import AudioSegment
import argparse
from pathlib import Path

def convert_m4a_to_audio(input_dir, output_dir=None, output_format="mp3", bitrate="128k"):
    """
    将指定目录中的所有.m4a文件转换为指定音频格式
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径，如果为None则使用输入目录
        output_format (str): 输出格式 (mp3, wav, aac, ogg)
        bitrate (str): 比特率，仅对压缩格式有效
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.m4a文件
    m4a_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not m4a_files:
        print(f"在目录 {input_dir} 中没有找到.m4a文件")
        return
    
    print(f"找到 {len(m4a_files)} 个.m4a文件")
    print(f"输出格式: {output_format.upper()}")
    if output_format in ["mp3", "aac", "ogg"]:
        print(f"比特率: {bitrate}")
    print("-" * 50)
    
    # 转换每个文件
    for i, m4a_file in enumerate(m4a_files, 1):
        try:
            # 获取文件名（不包含扩展名）
            filename = os.path.splitext(os.path.basename(m4a_file))[0]
            output_file = os.path.join(output_dir, f"{filename}.{output_format}")
            
            print(f"正在转换 ({i}/{len(m4a_files)}): {os.path.basename(m4a_file)}")
            
            # 加载音频文件
            audio = AudioSegment.from_file(m4a_file, format="wav")
            
            # 根据输出格式设置参数
            export_params = {"format": output_format}
            
            if output_format == "mp3":
                export_params["bitrate"] = bitrate
            elif output_format == "aac":
                export_params["bitrate"] = bitrate
            elif output_format == "ogg":
                export_params["bitrate"] = bitrate
            
            # 导出为指定格式
            audio.export(output_file, **export_params)
            
            # 显示文件大小信息
            original_size = os.path.getsize(m4a_file) / (1024 * 1024)  # MB
            new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            compression_ratio = (1 - new_size / original_size) * 100
            
            print(f"✓ 转换完成: {os.path.basename(output_file)}")
            print(f"  文件大小: {original_size:.1f}MB → {new_size:.1f}MB (压缩 {compression_ratio:.1f}%)")
            
        except Exception as e:
            print(f"✗ 转换失败 {os.path.basename(m4a_file)}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='将.m4a文件转换为多种音频格式')
    parser.add_argument('input_dir', nargs='?', default='samples', 
                        help='输入目录路径 (默认: samples)')
    parser.add_argument('-o', '--output', 
                        help='输出目录路径 (默认: 与输入目录相同)')
    parser.add_argument('-f', '--format', default='mp3',
                        choices=['mp3', 'wav', 'aac', 'ogg'],
                        help='输出格式 (默认: mp3)')
    parser.add_argument('-b', '--bitrate', default='128k',
                        choices=['96k', '128k', '192k', '256k', '320k'],
                        help='比特率，仅对压缩格式有效 (默认: 128k)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在")
        return
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output or args.input_dir}")
    print(f"输出格式: {args.format.upper()}")
    if args.format in ['mp3', 'aac', 'ogg']:
        print(f"比特率: {args.bitrate}")
    print("-" * 50)
    
    convert_m4a_to_audio(args.input_dir, args.output, args.format, args.bitrate)
    
    print("-" * 50)
    print("转换完成!")

if __name__ == "__main__":
    main() 