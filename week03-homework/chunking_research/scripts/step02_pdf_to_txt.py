#!/usr/bin/env python3
"""
PDF to Text Converter - 面向对象的PDF文本提取工具

功能说明：
========
本工具将PDF文件批量转换为文本文件，采用面向对象设计，具有以下特点：

1. 动态路径计算：
   - 自动基于脚本位置计算data/raw目录路径
   - 无论在哪个目录运行都能正确找到目标文件夹
   - 路径结构：scripts -> chunking_research -> data -> raw

2. 面向对象设计：
   - PDFToTextConverter类封装所有功能
   - 清晰的职责分离和单一职责原则
   - 支持灵活配置和自定义参数

3. 核心功能：
   - 批量转换：convert_all_pdfs() - 转换目录下所有PDF文件
   - 单文件转换：convert_single_pdf() - 转换指定PDF文件
   - 文件查找：find_pdf_files() - 查找所有PDF文件
   - 文本提取：extract_text_from_pdf() - 从PDF提取文本
   - 路径生成：generate_output_path() - 生成输出文件路径

使用方式：
========
# 基本使用（使用默认路径）
converter = PDFToTextConverter()
converter.convert_all_pdfs()

# 自定义路径
converter = PDFToTextConverter(
    input_dir="custom/input/path",
    output_dir="custom/output/path",
    encoding="utf-16"
)

# 转换单个文件
output_file = converter.convert_single_pdf("path/to/file.pdf")

# 查看配置信息
print(f"输入目录: {converter.input_dir}")
print(f"输出目录: {converter.output_dir}")

依赖要求：
========
pip install pymupdf

注意事项：
========
- 本工具兼容PyMuPDF 1.26.x版本的新API
- 自动遍历PDF的每一页进行文本提取
- 输出文件名格式：原文件名_txt.txt

作者: AI Assistant
日期: 2025-10-29
版本: 1.1.0 (修复PyMuPDF API兼容性问题)
"""

# pip install pymupdf

import fitz
import glob
import pathlib as P
from typing import List, Optional
import logging
import os


class PDFToTextConverter:
    """PDF到文本转换器类
    
    负责将PDF文件转换为文本文件的功能
    """
    
    @staticmethod
    def _get_default_data_dir() -> str:
        """获取默认的data/raw目录路径
        
        动态计算相对于当前脚本的data/raw目录路径，
        确保无论在哪里运行都能正确找到
        
        Returns:
            data/raw目录的绝对路径
        """
        # 获取当前脚本所在目录
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 从scripts目录向上找到chunking_research目录，然后定位data/raw
        # scripts -> chunking_research -> data -> raw
        chunking_research_dir = os.path.dirname(current_script_dir)
        data_raw_dir = os.path.join(chunking_research_dir, "data", "raw")
        
        return data_raw_dir
    
    def __init__(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None, 
                 encoding: str = "utf8"):
        """初始化PDF转换器
        
        Args:
            input_dir: PDF文件所在目录，默认为None（自动计算data/raw路径）
            output_dir: 输出文本文件目录，默认为None（与输入目录相同）
            encoding: 文本文件编码，默认为"utf8"
        """
        # 如果没有指定输入目录，则使用默认的data/raw目录
        if input_dir is None:
            input_dir = self._get_default_data_dir()
        
        self.input_dir = input_dir
        self.output_dir = output_dir or input_dir
        self.encoding = encoding
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"PDF转换器初始化完成")
        self.logger.info(f"输入目录: {self.input_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
    
    def find_pdf_files(self) -> List[str]:
        """查找输入目录中的所有PDF文件
        
        Returns:
            PDF文件路径列表
        """
        pattern = os.path.join(self.input_dir, "*.pdf")
        pdf_files = glob.glob(pattern)
        self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
            
        Raises:
            Exception: 当PDF文件无法打开或读取时
        """
        try:
            # 使用 fitz 打开PDF文档
            doc = fitz.open(pdf_path)  # type: ignore
            
            # 提取所有页面的文本
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # type: ignore
                page_text = page.get_text()  # type: ignore
                text_parts.append(page_text)
            
            # 合并所有页面的文本
            full_text = "\n".join(text_parts)
            
            doc.close()  # type: ignore
            return full_text
        except Exception as e:
            self.logger.error(f"无法从 {pdf_path} 提取文本: {e}")
            raise
    
    def generate_output_path(self, pdf_path: str) -> str:
        """生成输出文本文件路径
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            文本文件输出路径
        """
        filename = os.path.basename(pdf_path)
        txt_filename = filename.replace('.pdf', '_txt.txt')
        return os.path.join(self.output_dir, txt_filename)
    
    def save_text_to_file(self, text: str, output_path: str) -> None:
        """将文本保存到文件
        
        Args:
            text: 要保存的文本内容
            output_path: 输出文件路径
        """
        try:
            P.Path(output_path).write_text(text, encoding=self.encoding)
            self.logger.info(f"文本已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"无法保存文本到 {output_path}: {e}")
            raise
    
    def convert_single_pdf(self, pdf_path: str) -> str:
        """转换单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            输出文本文件路径
        """
        self.logger.info(f"正在转换: {pdf_path}")
        
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)
        
        # 生成输出路径
        output_path = self.generate_output_path(pdf_path)
        
        # 保存文本
        self.save_text_to_file(text, output_path)
        
        return output_path
    
    def convert_all_pdfs(self) -> List[str]:
        """转换输入目录中的所有PDF文件
        
        Returns:
            成功转换的文本文件路径列表
        """
        pdf_files = self.find_pdf_files()
        converted_files = []
        
        for pdf_path in pdf_files:
            try:
                output_path = self.convert_single_pdf(pdf_path)
                converted_files.append(output_path)
            except Exception as e:
                self.logger.error(f"转换失败 {pdf_path}: {e}")
                continue
        
        self.logger.info(f"成功转换 {len(converted_files)} 个文件")
        return converted_files


def main():
    """主函数 - 兼容原有的脚本行为"""
    converter = PDFToTextConverter()
    converter.convert_all_pdfs()


if __name__ == "__main__":
    main()
