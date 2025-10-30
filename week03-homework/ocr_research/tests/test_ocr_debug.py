#!/usr/bin/env python3
"""
OCR调试测试脚本
用于详细测试和调试OCR功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ocr_research.ImageOCRReader import ImageOCRReader

def test_ocr_environment():
    """测试OCR环境"""
    print("=" * 60)
    print("OCR环境测试")
    print("=" * 60)
    
    try:
        # 测试导入
        from paddleocr import PaddleOCR
        print("✓ PaddleOCR导入成功")
        
        # 测试基本初始化
        print("\n测试OCR初始化...")
        reader = ImageOCRReader(lang='ch', use_gpu=False)
        print("✓ OCR初始化成功")
        
        return reader
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请安装PaddleOCR: pip install paddleocr")
        return None
    except Exception as e:
        print(f"✗ 初始化错误: {e}")
        return None

def test_with_sample_image(reader):
    """使用示例图像测试"""
    print("\n" + "=" * 60)
    print("示例图像测试")
    print("=" * 60)
    
    # 查找data目录下的图像文件
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"✗ 数据目录不存在: {data_dir}")
        print("请在 ocr_research/data/ 目录下放置一些测试图像")
        return
    
    # 查找图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f"*{ext}"))
        image_files.extend(data_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"✗ 在 {data_dir} 中未找到图像文件")
        print(f"支持的格式: {', '.join(image_extensions)}")
        return
    
    # 测试第一个图像文件
    test_image = str(image_files[0])
    print(f"测试图像: {test_image}")
    
    try:
        # 使用简单测试方法
        success = reader.test_ocr_simple(test_image)
        if success:
            print("✓ OCR简单测试通过")
        else:
            print("✗ OCR简单测试失败")
            
        # 使用完整的load_data方法
        print("\n使用完整load_data方法测试...")
        documents = reader.load_data(test_image)
        
        if documents:
            doc = documents[0]
            print(f"✓ 成功创建Document")
            print(f"文本长度: {len(doc.text)}")
            print(f"元数据: {doc.metadata}")
            print(f"文本预览: {doc.text[:200]}...")
        else:
            print("✗ 未生成Document")
            
    except Exception as e:
        print(f"✗ 测试失败: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    print("PaddleOCR调试测试")
    print("当前工作目录:", os.getcwd())
    print("Python路径:", sys.path[:3])
    
    # 环境测试
    reader = test_ocr_environment()
    if reader is None:
        print("环境测试失败，退出")
        return
    
    # 图像测试
    test_with_sample_image(reader)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()