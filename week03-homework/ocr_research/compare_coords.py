#!/usr/bin/env python3
"""
比较坐标修复前后的可视化效果
"""

import os
from ImageOCRReader import ImageOCRReader

def main():
    # 初始化OCR读取器
    reader = ImageOCRReader()
    
    # 输入图像路径
    image_path = "ocr_research/data/natural_scene/test.jpeg"
    
    print("=== 坐标修复效果对比 ===")
    print(f"输入图像: {image_path}")
    
    # 生成修复后的可视化
    output_path = "ocr_research/output/test_fixed_coordinates.jpg"
    result = reader.visualize_ocr(image_path, output_path)
    
    print(f"\n✅ 修复后的可视化已保存到: {result}")
    
    # 显示文件信息
    if os.path.exists(result):
        file_size = os.path.getsize(result)
        print(f"📊 文件大小: {file_size:,} bytes")
    
    print("\n=== 修复要点 ===")
    print("1. 🔧 正确的坐标转换链: OCR坐标 -> 原图坐标 -> 可视化坐标")
    print("2. 📏 考虑OCR内部缩放比例 (1500px限制)")
    print("3. 🖼️  考虑可视化缩放比例 (1024px限制)")
    print("4. 🎯 精确的多重缩放计算")
    print("5. 🔍 详细的调试日志输出")
    
    # 检查文本识别结果
    docs = reader.load_data([image_path])
    if docs:
        text_content = docs[0].text
        print(f"\n📝 识别的文本内容:")
        print(f"   总字符数: {len(text_content)}")
        lines = text_content.strip().split('\n')
        print(f"   文本行数: {len(lines)}")
        for i, line in enumerate(lines[:5], 1):  # 显示前5行
            print(f"   {i}. {line}")
        if len(lines) > 5:
            print(f"   ... (还有{len(lines)-5}行)")

if __name__ == "__main__":
    main()