
try:
    from .ImageOCRReader import ImageOCRReader
except ImportError:
    from ImageOCRReader import ImageOCRReader

# 加载环境变量
import os
from pathlib import Path


def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m ocr_research.main 运行
    #pass

    try:
        reader = ImageOCRReader(lang='ch')
        
        
        test_files = [
            "ocr_research/data/natural_scene/test.jpeg",
            # "ocr_research/data/scanned_docs/test.jpeg",  # 先注释掉其他文件
            # "ocr_research/data/screenshots/test.jpeg"
        ]
        
        print(f"准备处理 {len(test_files)} 个文件")
        documents = reader.load_data(test_files)
        
        print(f"成功处理 {len(documents)} 个文档")

        # 框选文字可视化
        reader.visualize_ocr(test_files[0],output_path="ocr_research/output/test_annotated.png")
        
        
    except Exception as e:
        print(f"程序执行失败: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()