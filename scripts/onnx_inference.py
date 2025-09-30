#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型推理脚本
提供基于ONNX Runtime的高效推理功能
"""

import os
import time
import numpy as np
import cv2
from PIL import Image
import argparse
import json
from typing import List, Tuple, Union, Optional

try:
    import onnxruntime as ort
except ImportError:
    print("请安装onnxruntime: pip install onnxruntime")
    exit(1)


class ONNXPredictor:
    """
    ONNX模型推理器
    """
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        初始化ONNX推理器
        
        Args:
            model_path (str): ONNX模型文件路径
            providers (List[str], optional): 推理提供者列表，默认为['CPUExecutionProvider']
        """
        self.model_path = model_path
        
        # 设置推理提供者
        if providers is None:
            # 优先使用GPU，如果不可用则使用CPU
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        self.providers = providers
        
        # 创建推理会话
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"成功加载ONNX模型: {model_path}")
            print(f"使用推理提供者: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"输入节点: {self.input_name}, 形状: {self.input_shape}")
        print(f"输出节点: {self.output_name}, 形状: {self.output_shape}")
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        预处理输入图像
        
        Args:
            image_path (str): 图像文件路径
            target_size (Tuple[int, int]): 目标尺寸
            
        Returns:
            np.ndarray: 预处理后的图像张量
        """
        try:
            # 读取图像
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"无法读取图像: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 如果输入是numpy数组
                image = image_path
            
            # 调整尺寸
            image = cv2.resize(image, target_size)
            
            # 归一化到[0,1]
            image = image.astype(np.float32) / 255.0
            
            # 标准化 (ImageNet统计值)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # 转换维度: HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
            
            # 添加batch维度: CHW -> NCHW
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"图像预处理失败: {e}")
    
    def predict(self, input_data: Union[str, np.ndarray], return_raw: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        执行模型推理
        
        Args:
            input_data (Union[str, np.ndarray]): 输入数据，可以是图像路径或预处理后的数组
            return_raw (bool): 是否返回原始输出
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, float]]: 推理结果或(推理结果, 推理时间)
        """
        # 预处理输入
        if isinstance(input_data, str):
            processed_input = self.preprocess_image(input_data)
        else:
            processed_input = input_data
        
        # 执行推理
        start_time = time.time()
        try:
            outputs = self.session.run([self.output_name], {self.input_name: processed_input})
            inference_time = time.time() - start_time
            
            output = outputs[0]
            
            if return_raw:
                return output, inference_time
            else:
                return output
                
        except Exception as e:
            raise RuntimeError(f"模型推理失败: {e}")
    
    def predict_batch(self, input_list: List[Union[str, np.ndarray]]) -> Tuple[np.ndarray, List[float]]:
        """
        批量推理
        
        Args:
            input_list (List[Union[str, np.ndarray]]): 输入列表
            
        Returns:
            Tuple[np.ndarray, List[float]]: (批量推理结果, 每个样本的推理时间)
        """
        results = []
        inference_times = []
        
        for input_data in input_list:
            output, inference_time = self.predict(input_data, return_raw=True)
            results.append(output)
            inference_times.append(inference_time)
        
        # 将结果合并为批次
        batch_results = np.vstack(results) if results else np.array([])
        
        return batch_results, inference_times
    
    def get_feature_embedding(self, input_data: Union[str, np.ndarray]) -> np.ndarray:
        """
        获取特征向量（用于相似度比较）
        
        Args:
            input_data (Union[str, np.ndarray]): 输入数据
            
        Returns:
            np.ndarray: 特征向量
        """
        output = self.predict(input_data)
        # 对特征向量进行L2归一化
        feature = output / np.linalg.norm(output, axis=1, keepdims=True)
        return feature
    
    def calculate_similarity(self, image1: Union[str, np.ndarray], 
                           image2: Union[str, np.ndarray], 
                           metric: str = 'cosine') -> float:
        """
        计算两张图像的相似度
        
        Args:
            image1, image2: 输入图像
            metric (str): 相似度度量方法，支持'cosine'和'euclidean'
            
        Returns:
            float: 相似度分数
        """
        # 获取特征向量
        feat1 = self.get_feature_embedding(image1)
        feat2 = self.get_feature_embedding(image2)
        
        if metric == 'cosine':
            # 余弦相似度
            similarity = np.dot(feat1.flatten(), feat2.flatten())
        elif metric == 'euclidean':
            # 欧氏距离（转换为相似度）
            distance = np.linalg.norm(feat1 - feat2)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"不支持的相似度度量方法: {metric}")
        
        return float(similarity)
    
    def benchmark(self, input_data: Union[str, np.ndarray], num_runs: int = 100) -> dict:
        """
        性能基准测试
        
        Args:
            input_data: 测试输入数据
            num_runs (int): 运行次数
            
        Returns:
            dict: 基准测试结果
        """
        print(f"开始性能基准测试，运行 {num_runs} 次...")
        
        # 预处理输入
        if isinstance(input_data, str):
            processed_input = self.preprocess_image(input_data)
        else:
            processed_input = input_data
        
        # 预热
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: processed_input})
        
        # 正式测试
        times = []
        for i in range(num_runs):
            start_time = time.time()
            self.session.run([self.output_name], {self.input_name: processed_input})
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{num_runs} 次推理")
        
        times = np.array(times)
        
        results = {
            'num_runs': num_runs,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'fps': 1.0 / np.mean(times),
            'throughput_per_second': 1.0 / np.mean(times)
        }
        
        return results


def main():
    # 简单变量赋值替代命令行参数
    model_path = "/data/apps/wal/signature_detector/checkpoints/model-acc-122-0.9559-0.9783-0.9626.onnx"  # ONNX模型文件路径
    input_path = "/data/apps/wal/signature_detector/data/imgs/1754030710565.png"  # 输入图像路径或图像目录
    output_path = None  # 输出结果保存路径
    run_benchmark = False  # 是否运行性能基准测试
    similarity_paths = None  # 计算两张图像的相似度，提供两个图像路径
    providers = None  # 指定推理提供者，如['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: ONNX模型文件不存在: {model_path}")
        return
    
    # 创建推理器
    try:
        predictor = ONNXPredictor(model_path, providers=providers)
    except Exception as e:
        print(f"初始化推理器失败: {e}")
        return
    
    # 相似度计算模式
    if similarity_paths:
        img1, img2 = similarity_paths
        if not os.path.exists(img1) or not os.path.exists(img2):
            print("错误: 图像文件不存在")
            return
        
        try:
            similarity = predictor.calculate_similarity(img1, img2)
            print(f"图像相似度: {similarity:.4f}")
        except Exception as e:
            print(f"相似度计算失败: {e}")
        return
    
    # 性能测试模式
    if run_benchmark:
        if not os.path.exists(input_path):
            print(f"错误: 输入文件不存在: {input_path}")
            return
        
        try:
            results = predictor.benchmark(input_path)
            print("性能基准测试结果:")
            print(f"平均推理时间: {results['mean_time']:.4f} 秒")
            print(f"标准差: {results['std_time']:.4f} 秒")
            print(f"最小时间: {results['min_time']:.4f} 秒")
            print(f"最大时间: {results['max_time']:.4f} 秒")
            print(f"FPS: {results['fps']:.2f}")
            
            if output_path:
                with open(output_path, 'w', encoding='utf8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"结果已保存至: {output_path}")
                
        except Exception as e:
            print(f"性能测试失败: {e}")
        return
    
    # 推理模式
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    try:
        if os.path.isfile(input_path):
            # 单个文件推理
            output, inference_time = predictor.predict(input_path, return_raw=True)
            print(f"推理完成，耗时: {inference_time:.4f} 秒")
            print(f"输出形状: {output.shape}")
            print(f"特征向量: {output.flatten()[:10]}...")  # 显示前10个值
            
            if output_path:
                np.save(output_path, output)
                print(f"结果已保存至: {output_path}")
        
        elif os.path.isdir(input_path):
            # 批量推理
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(input_path, ext)))
                image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
            
            if not image_files:
                print("错误: 未找到图像文件")
                return
            
            print(f"找到 {len(image_files)} 个图像文件")
            
            batch_results, inference_times = predictor.predict_batch(image_files)
            
            print(f"批量推理完成")
            print(f"平均推理时间: {np.mean(inference_times):.4f} 秒")
            print(f"输出形状: {batch_results.shape}")
            
            if output_path:
                result_dict = {
                    'files': image_files,
                    'features': batch_results.tolist(),
                    'inference_times': inference_times
                }
                with open(output_path, 'w', encoding='utf8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                print(f"结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"推理失败: {e}")


if __name__ == '__main__':
    import glob
    main()