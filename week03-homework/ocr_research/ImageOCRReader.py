# ocr_research/reader.py
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from paddleocr import PaddleOCR
from typing import List, Union, Dict, Any, Optional
import os
import cv2
import numpy as np
from pathlib import Path
import json
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageOCRReader(BaseReader):
    """使用PP-OCR从图像中提取文本并返回Document"""
    
    def __init__(self, 
                 lang='ch',
                 use_gpu=False,
                 ocr_version='PP-OCRv5',
                 text_detection_model_name=None,
                 text_recognition_model_name=None,
                 **kwargs):
        """
        Args:
            lang: OCR语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用GPU加速
            ocr_version: OCR版本
            **kwargs: 其他传递给PaddleOCR的参数
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version
        
        # 初始化OCR模型
        ocr_params = {
            'lang': lang,
            'ocr_version': ocr_version,
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
            **kwargs
        }
        
        # 新版本 PaddleOCR 使用 device 参数而不是 use_gpu
        if use_gpu:
            ocr_params['device'] = 'gpu'
        else:
            ocr_params['device'] = 'cpu'
        
        # 如果指定了特定模型
        if text_detection_model_name:
            ocr_params['text_detection_model_name'] = text_detection_model_name
        if text_recognition_model_name:
            ocr_params['text_recognition_model_name'] = text_recognition_model_name
        
        print(f"[OCR INIT] 初始化OCR参数: {ocr_params}")
        
        try:
            self.ocr = PaddleOCR(**ocr_params)
            print(f"[OCR INIT] OCR模型初始化成功")
        except Exception as init_error:
            print(f"[OCR INIT ERROR] OCR模型初始化失败: {type(init_error).__name__}: {str(init_error)}")
            import traceback
            traceback.print_exc()
            raise init_error
    
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回Document列表
        Args:
            file: 图像路径字符串或路径列表
        Returns:
            List[Document]
        """
        if isinstance(file, str):
            files = [file]
        else:
            files = file
            
        print(f"[OCR DEBUG] 准备处理 {len(files)} 个文件")
        documents = []
        
        for i, file_path in enumerate(files):
            print(f"[OCR DEBUG] 处理文件 {i+1}/{len(files)}: {file_path}")
            
            # 将路径标准化为绝对路径
            abs_file_path = os.path.abspath(file_path)
            print(f"[OCR DEBUG] 绝对路径: {abs_file_path}")
            
            if not os.path.exists(abs_file_path):
                error_msg = f"图像文件不存在: {abs_file_path}"
                print(f"[OCR ERROR] {error_msg}")
                raise FileNotFoundError(error_msg)
            
        try:
            print(f"[OCR DEBUG] 开始OCR处理...")
            ocr_results = self._perform_ocr(abs_file_path)
            
            # 创建Document
            print(f"[OCR DEBUG] 创建Document...")
            doc = self._create_document(abs_file_path, ocr_results)
            documents.append(doc)
            print(f"[OCR DEBUG] 文件处理完成")
            
        except Exception as file_error:
            print(f"[OCR ERROR] 处理文件 {file_path} 时发生错误: {type(file_error).__name__}: {str(file_error)}")
            import traceback
            traceback.print_exc()
            raise file_error
        finally:
            # 清理临时文件
            temp_dir = os.path.dirname(abs_file_path)
            temp_name = f"temp_resized_{os.path.basename(abs_file_path)}"
            temp_path = os.path.join(temp_dir, temp_name)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[OCR DEBUG] 清理临时文件: {temp_path}")
                except Exception as cleanup_error:
                    print(f"[OCR DEBUG] 清理临时文件失败: {cleanup_error}")
        
        print(f"[OCR DEBUG] 所有文件处理完成，共生成 {len(documents)} 个Document")
        return documents
    
    def _perform_ocr(self, image_path: str) -> Dict[str, Any]:
        """执行OCR并返回结构化结果"""
        try:
            print(f"[OCR DEBUG] 开始处理图像: {image_path}")
            print(f"[OCR DEBUG] 图像文件存在: {os.path.exists(image_path)}")
            print(f"[OCR DEBUG] 图像文件大小: {os.path.getsize(image_path)} bytes")
            
            # 检查图像文件是否可读并获取尺寸信息
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    print(f"[OCR DEBUG] 图像尺寸: {img.shape}")
                    
                    # 检查图像是否过大，如果是则进行缩放
                    max_dimension = 1500  # 降低最大尺寸限制，避免内存问题
                    if max(height, width) > max_dimension:
                        print(f"[OCR DEBUG] 图像过大({width}x{height})，进行缩放...")
                        scale = max_dimension / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                        print(f"[OCR DEBUG] 缩放后尺寸: {new_width}x{new_height}")
                        
                        # 保存缩放后的临时图像
                        temp_dir = os.path.dirname(image_path)
                        temp_name = f"temp_resized_{os.path.basename(image_path)}"
                        temp_path = os.path.join(temp_dir, temp_name)
                        cv2.imwrite(temp_path, img)
                        image_path = temp_path  # 使用缩放后的图像
                        print(f"[OCR DEBUG] 临时文件: {temp_path}")
                    else:
                        print(f"[OCR DEBUG] 图像尺寸适中: {width}x{height}")
                else:
                    print(f"[OCR DEBUG] 警告: cv2无法读取图像文件")
            except Exception as img_check_error:
                print(f"[OCR DEBUG] 图像检查失败: {img_check_error}")
            
            print(f"[OCR DEBUG] OCR参数 - 语言: {self.lang}, 版本: {self.ocr_version}")
            print(f"[OCR DEBUG] 调用 OCR predict...")
            
            # 添加内存清理
            import gc
            gc.collect()
            
            results = self.ocr.predict(image_path)
            
            print(f"[OCR DEBUG] OCR执行完成")
            print(f"[OCR DEBUG] 结果类型: {type(results)}")
            print(f"[OCR DEBUG] 结果长度: {len(results) if results else 0}")
            if results:
                print(f"[OCR DEBUG] 第一层结果类型: {type(results[0])}")
                print(f"[OCR DEBUG] 第一层结果长度: {len(results[0]) if results[0] else 0}")
            
        except Exception as ocr_error:
            print(f"[OCR ERROR] OCR执行失败: {type(ocr_error).__name__}: {str(ocr_error)}")
            import traceback
            print(f"[OCR ERROR] 详细错误堆栈:")
            traceback.print_exc()
            raise ocr_error
        
        text_blocks = []
        total_confidence = 0
        num_blocks = 0
        
        try:
            if results and results[0]:
                print(f"[OCR DEBUG] 开始解析OCR结果...")
                result_obj = results[0]
                
                # 检查结果对象的类型和属性
                print(f"[OCR DEBUG] 结果对象类型: {type(result_obj)}")
                
                # 新版本PaddleOCR返回OCRResult对象（类似字典）
                if 'rec_texts' in result_obj and 'rec_scores' in result_obj:
                    print(f"[OCR DEBUG] 使用新版本OCRResult格式...")
                    rec_texts = result_obj['rec_texts']
                    rec_scores = result_obj['rec_scores']
                    rec_polys = result_obj.get('rec_polys', [])
                    
                    print(f"[OCR DEBUG] 检测到 {len(rec_texts)} 个文本")
                    
                    for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                        print(f"[OCR DEBUG] 文本块 {i+1}: '{text}' (置信度: {score:.3f})")
                        
                        # 获取边界框信息
                        bbox = rec_polys[i] if rec_polys and i < len(rec_polys) else []
                        
                        text_blocks.append({
                            'text': text,
                            'confidence': float(score),
                            'bbox': bbox
                        })
                        total_confidence += float(score)
                        num_blocks += 1
                        
                # 如果有rec_texts属性而不是字典键
                elif hasattr(result_obj, 'rec_texts') and hasattr(result_obj, 'rec_scores'):
                    print(f"[OCR DEBUG] 使用属性访问方式...")
                    rec_texts = result_obj.rec_texts
                    rec_scores = result_obj.rec_scores
                    rec_polys = getattr(result_obj, 'rec_polys', [])
                    
                    print(f"[OCR DEBUG] 检测到 {len(rec_texts)} 个文本")
                    
                    for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                        print(f"[OCR DEBUG] 文本块 {i+1}: '{text}' (置信度: {score:.3f})")
                        
                        # 获取边界框信息
                        bbox = rec_polys[i] if rec_polys and i < len(rec_polys) else []
                        
                        text_blocks.append({
                            'text': text,
                            'confidence': float(score),
                            'bbox': bbox
                        })
                        total_confidence += float(score)
                        num_blocks += 1
                        
                # 尝试旧版本格式 - 暂时禁用，先查看对象结构
                elif False and hasattr(result_obj, '__iter__'):
                    print(f"[OCR DEBUG] 尝试旧版本格式...")
                    for i, line in enumerate(result_obj):
                        if line and len(line) == 2:
                            try:
                                bbox, (text, confidence) = line
                                print(f"[OCR DEBUG] 文本块 {i+1}: '{text}' (置信度: {confidence:.3f})")
                                text_blocks.append({
                                    'text': text,
                                    'confidence': float(confidence),
                                    'bbox': bbox
                                })
                                total_confidence += float(confidence)
                                num_blocks += 1
                            except Exception as parse_error:
                                print(f"[OCR ERROR] 解析第{i+1}个文本块失败: {parse_error}")
                                print(f"[OCR ERROR] 问题行数据: {line}")
                else:
                    print(f"[OCR DEBUG] 未知的结果格式，尝试直接访问属性...")
                    # 打印可用属性
                    attrs = [attr for attr in dir(result_obj) if not attr.startswith('_')]
                    print(f"[OCR DEBUG] 可用属性: {attrs}")
                    
                    # 尝试访问一些可能的属性
                    if hasattr(result_obj, 'text'):
                        print(f"[OCR DEBUG] 找到text属性: {result_obj.text}")
                    if hasattr(result_obj, 'texts'):
                        print(f"[OCR DEBUG] 找到texts属性: {result_obj.texts}")
                    if hasattr(result_obj, 'boxes'):
                        print(f"[OCR DEBUG] 找到boxes属性: {result_obj.boxes}")
                    
                    # 尝试通过字典方式访问
                    try:
                        if hasattr(result_obj, 'keys'):
                            print(f"[OCR DEBUG] 对象可能是字典类型，keys: {list(result_obj.keys())}")
                    except:
                        pass
            else:
                print(f"[OCR DEBUG] OCR结果为空或无效")
                
        except Exception as result_parse_error:
            print(f"[OCR ERROR] 结果解析失败: {result_parse_error}")
            import traceback
            traceback.print_exc()
        
        avg_confidence = total_confidence / num_blocks if num_blocks > 0 else 0
        
        print(f"[OCR DEBUG] 解析完成 - 共{num_blocks}个文本块，平均置信度: {avg_confidence:.3f}")
        
        return {
            'text_blocks': text_blocks,
            'num_text_blocks': num_blocks,
            'avg_confidence': avg_confidence
        }
    
    def _create_document(self, image_path: str, ocr_results: Dict[str, Any]) -> Document:
        """根据OCR结果创建Document对象"""
        # 拼接文本
        full_text = self._concatenate_text(ocr_results['text_blocks'])
        
        # 准备元数据
        metadata = {
            'image_path': str(image_path),
            'ocr_model': self.ocr_version,
            'language': self.lang,
            'num_text_blocks': ocr_results['num_text_blocks'],
            'avg_confidence': ocr_results['avg_confidence'],
            'file_name': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path)
        }
        
        return Document(
            text=full_text,
            metadata=metadata
        )
    
    def _concatenate_text(self, text_blocks: List[Dict]) -> str:
        """将文本块拼接成完整文本"""
        # 按垂直位置排序（假设从上到下阅读）
        sorted_blocks = sorted(text_blocks, key=lambda x: x['bbox'][0][1])
        
        # 拼接文本，保留一定的结构
        lines = []
        for block in sorted_blocks:
            text = block['text'].strip()
            conf = block['confidence']
            lines.append(f"[Text Block] (conf: {conf:.2f}): {text}")
        
        return '\n'.join(lines)
    
    def visualize_ocr(self, image_path: str, output_path: Optional[str] = None) -> str:
        """可视化OCR结果 - 修复坐标映射问题"""
        if output_path is None:
            output_path = f"{os.path.splitext(image_path)[0]}_ocr_vis.jpg"
            
        # 将路径标准化为绝对路径
        abs_image_path = os.path.abspath(image_path)
        abs_output_path = os.path.abspath(output_path)
        
        print(f"[OCR VIS] 开始可视化处理")
        print(f"[OCR VIS] 输入图像: {abs_image_path}")
        print(f"[OCR VIS] 输出路径: {abs_output_path}")
        
        try:
            # 创建输出目录
            output_dir = os.path.dirname(abs_output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"[OCR VIS] 创建输出目录: {output_dir}")
            
            # 读取原始图像
            print(f"[OCR VIS] 读取原始图像...")
            original_img = cv2.imread(abs_image_path)
            if original_img is None:
                raise ValueError(f"无法读取图像: {abs_image_path}")
            
            # 获取原始图像尺寸
            original_height, original_width = original_img.shape[:2]
            print(f"[OCR VIS] 原始图像尺寸: {original_width}x{original_height}")
            
            # 准备可视化图像（缩放用于显示）
            vis_max_dimension = 1024
            if max(original_height, original_width) > vis_max_dimension:
                vis_scale = vis_max_dimension / max(original_height, original_width)
                vis_width = int(original_width * vis_scale)
                vis_height = int(original_height * vis_scale)
                vis_img = cv2.resize(original_img, (vis_width, vis_height))
                print(f"[OCR VIS] 可视化图像缩放到: {vis_width}x{vis_height} (比例: {vis_scale:.3f})")
            else:
                vis_img = original_img.copy()
                vis_scale = 1.0
                print(f"[OCR VIS] 使用原始尺寸进行可视化")
            
            # 直接对原始图像执行OCR，获取基于原始坐标系的结果
            print(f"[OCR VIS] 对原始图像执行OCR...")
            
            # 如果原图太大，我们需要知道OCR内部的缩放比例
            ocr_max_dimension = 1500  # 这是_perform_ocr中使用的最大尺寸
            if max(original_height, original_width) > ocr_max_dimension:
                ocr_scale = ocr_max_dimension / max(original_height, original_width)
                print(f"[OCR VIS] OCR内部会缩放图像，比例: {ocr_scale:.3f}")
            else:
                ocr_scale = 1.0
                print(f"[OCR VIS] OCR使用原始尺寸")
            
            # 执行OCR
            results = self._perform_ocr(abs_image_path)
            print(f"[OCR VIS] OCR执行完成，结果类型: {type(results)}")
            
            # 绘制结果
            text_drawn = 0
            if results and 'text_blocks' in results:
                text_blocks = results['text_blocks']
                print(f"[OCR VIS] 找到 {len(text_blocks)} 个文本区域")
                
                for i, block in enumerate(text_blocks):
                    text = block.get('text', '')
                    confidence = block.get('confidence', 0.0)
                    bbox = block.get('bbox', [])
                    
                    if bbox is not None and len(bbox) >= 4:
                        try:
                            # 坐标转换：OCR坐标 -> 原图坐标 -> 可视化坐标
                            # OCR坐标基于OCR内部缩放后的图像，需要先还原到原图，再缩放到可视化尺寸
                            
                            # 第一步：从OCR坐标还原到原图坐标
                            if ocr_scale != 1.0:
                                # OCR内部缩放了，坐标需要放大回原图
                                original_poly = [[p[0] / ocr_scale, p[1] / ocr_scale] for p in bbox]
                            else:
                                original_poly = bbox
                            
                            # 第二步：从原图坐标缩放到可视化坐标
                            vis_poly = [[int(p[0] * vis_scale), int(p[1] * vis_scale)] for p in original_poly]
                            
                            print(f"[OCR VIS] 文本{i+1} 坐标转换: OCR{bbox[0]} -> 原图{[int(original_poly[0][0]), int(original_poly[0][1])]} -> 可视化{vis_poly[0]}")
                            
                            pts = np.array(vis_poly, np.int32)
                            
                            # 画边界框
                            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)
                            
                            # 添加文本标签
                            if len(vis_poly) > 0:
                                label = f"{text[:15]} ({confidence:.2f})"
                                # 文本位置稍微上移
                                text_y = max(vis_poly[0][1] - 15, 25)
                                cv2.putText(vis_img, label, (vis_poly[0][0], text_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            text_drawn += 1
                            print(f"[OCR VIS] 成功绘制文本 {i+1}: {text[:20]}")
                            
                        except Exception as draw_error:
                            print(f"[OCR VIS] 绘制文本 {i+1} 失败: {draw_error}")
                            import traceback
                            traceback.print_exc()
            else:
                print(f"[OCR VIS] 警告: OCR结果为空或格式不正确")
            
            print(f"[OCR VIS] 共绘制了 {text_drawn} 个文本框")
            
            # 保存图像
            print(f"[OCR VIS] 保存图像到: {abs_output_path}")
            success = cv2.imwrite(abs_output_path, vis_img)
            if not success:
                raise RuntimeError(f"无法保存图像到: {abs_output_path}")
            
            # 验证文件是否成功保存
            if os.path.exists(abs_output_path):
                file_size = os.path.getsize(abs_output_path)
                print(f"[OCR VIS] 保存成功! 文件大小: {file_size} bytes")
            else:
                raise RuntimeError(f"文件保存失败，文件不存在: {abs_output_path}")
                
            return abs_output_path
            
        except Exception as e:
            print(f"[OCR VIS ERROR] 可视化失败: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def test_ocr_simple(self, image_path: str) -> bool:
        """简单测试OCR是否正常工作"""
        try:
            print(f"[OCR TEST] 测试图像: {image_path}")
            abs_path = os.path.abspath(image_path)
            
            if not os.path.exists(abs_path):
                print(f"[OCR TEST ERROR] 文件不存在: {abs_path}")
                return False
            
            print(f"[OCR TEST] 调用OCR predict...")
            results = self.ocr.predict(abs_path)
            
            print(f"[OCR TEST] OCR调用成功")
            print(f"[OCR TEST] 结果类型: {type(results)}")
            print(f"[OCR TEST] 结果内容: {results}")
            
            if results and len(results) > 0 and results[0]:
                print(f"[OCR TEST] 检测到 {len(results[0])} 个文本区域")
                for i, line in enumerate(results[0][:3]):  # 只显示前3个
                    if line:
                        bbox, (text, confidence) = line
                        print(f"[OCR TEST] 文本{i+1}: '{text}' (置信度: {confidence:.3f})")
                return True
            else:
                print(f"[OCR TEST] 未检测到文本")
                return True  # OCR工作正常，只是没有检测到文本
                
        except Exception as e:
            print(f"[OCR TEST ERROR] OCR测试失败: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False