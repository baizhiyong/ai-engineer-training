# 安装依赖:
# pip install pdf2image Pillow
# pip install pdf2image --index-url https://pypi.org/simple/ 
# macOS 系统依赖:
# brew install poppler
#
# Ubuntu/Debian 系统依赖:
# sudo apt-get install poppler-utils
#
# Windows 系统依赖:
# 需要手动下载 poppler 并配置 PATH，或使用 conda:
# conda install -c conda-forge poppler

import os
import json
from pathlib import Path
from typing import List
import pdf2image
from PIL import Image

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """将PDF转换为图像列表"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    
    for i, image in enumerate(images):
        image_file_path = output_path / f"page_{i+1}.jpg"
        image.save(str(image_file_path), 'JPEG')
        image_paths.append(str(image_file_path))
    
    return image_paths

def batch_process_images(reader, input_dir: str, output_dir: str) -> List[str]:
    """批量处理图像目录"""
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    documents = reader.load_data([str(f) for f in image_files])
    
    # 保存处理结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    for i, doc in enumerate(documents):
        result_file = output_path / f"result_{i+1}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'text': doc.text,
                'metadata': doc.metadata
            }, f, ensure_ascii=False, indent=2)
        saved_files.append(str(result_file))
    
    return saved_files