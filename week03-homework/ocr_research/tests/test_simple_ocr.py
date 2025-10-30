#!/usr/bin/env python3
"""
简单的OCR测试脚本，不依赖LlamaIndex
用于排查bus error问题
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_paddleocr_directly():
    """直接测试PaddleOCR，不通过我们的封装"""
    print("=" * 60)
    print("直接测试PaddleOCR")
    print("=" * 60)
    
    try:
        from paddleocr import PaddleOCR
        
        # 最简单的初始化
        print("初始化PaddleOCR...")
        ocr = PaddleOCR(
            lang='ch', 
            device='cpu',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        print("✓ PaddleOCR初始化成功")
        
        # 测试图像路径
        test_image = "data/natural_scene/test.jpeg"
        abs_path = os.path.abspath(test_image)
        
        if not os.path.exists(abs_path):
            print(f"✗ 测试图像不存在: {abs_path}")
            return False
            
        print(f"测试图像: {abs_path}")
        file_size = os.path.getsize(abs_path)
        print(f"文件大小: {file_size} bytes")
        
        # 如果文件太大，先检查图像尺寸
        if file_size > 500000:  # 500KB
            print("文件较大，检查图像尺寸...")
            try:
                import cv2
                img = cv2.imread(abs_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"图像尺寸: {width}x{height}")
                    
                    # 如果图像过大，缩放它
                    if max(height, width) > 1024:
                        print("图像过大，进行缩放...")
                        scale = 1024 / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img_resized = cv2.resize(img, (new_width, new_height))
                        
                        # 保存临时文件
                        temp_path = abs_path.replace('.jpeg', '_small.jpeg')
                        cv2.imwrite(temp_path, img_resized)
                        abs_path = temp_path
                        print(f"使用缩放后的图像: {abs_path}")
                        
            except Exception as e:
                print(f"图像缩放失败: {e}")
        
        print("开始OCR处理...")
        
        # 执行OCR
        results = ocr.predict(abs_path)
        
        print("✓ OCR处理完成")
        print(f"结果类型: {type(results)}")
        
        if results and results[0]:
            print(f"检测到 {len(results[0])} 个文本区域")
            for i, line in enumerate(results[0][:3]):  # 只显示前3个
                if line:
                    bbox, (text, confidence) = line
                    print(f"文本{i+1}: '{text}' (置信度: {confidence:.3f})")
        else:
            print("未检测到文本")
            
        return True
        
    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_our_reader():
    """测试我们的封装类"""
    print("\n" + "=" * 60)
    print("测试我们的ImageOCRReader")
    print("=" * 60)
    
    try:
        from ocr_research.ImageOCRReader import ImageOCRReader
        
        print("初始化ImageOCRReader...")
        reader = ImageOCRReader(lang='ch', use_gpu=False)
        
        print("测试simple方法...")
        success = reader.test_ocr_simple("data/natural_scene/test.jpeg")
        
        if success:
            print("✓ 简单测试通过")
        else:
            print("✗ 简单测试失败")
            
        return success
        
    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("OCR Bus Error 调试")
    print("当前工作目录:", os.getcwd())
    
    # 切换到正确的目录
    os.chdir("/Users/baizhiyong/vscode/ai-engineering-practices/week03-homework/ocr_research")
    print("切换到目录:", os.getcwd())
    
    # 1. 直接测试PaddleOCR
    success1 = test_paddleocr_directly()
    
    if success1:
        # 2. 测试我们的封装
        test_with_our_reader()
    else:
        print("PaddleOCR基础测试失败，跳过封装测试")
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()