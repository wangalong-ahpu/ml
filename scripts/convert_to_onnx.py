#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch模型转ONNX格式脚本
支持将训练好的PyTorch模型转换为ONNX格式，方便部署和推理
"""

import os
import torch
import torch.onnx
import argparse
import yaml
from models.backbone_attention import Backbone


def convert_model_to_onnx(model_path, output_path, model_config, input_size=(1, 3, 224, 224)):
    """
    将PyTorch模型转换为ONNX格式
    
    Args:
        model_path (str): PyTorch模型文件路径
        output_path (str): 输出ONNX模型路径
        model_config (dict): 模型配置信息
        input_size (tuple): 输入张量尺寸，默认为(1, 3, 224, 224)
    """
    
    # 提取配置参数
    out_dimension = model_config.get("out_dimension", 256)
    model_name = model_config.get("model_name", "resnet18")
    
    print(f"正在转换模型: {model_path}")
    print(f"模型架构: {model_name}")
    print(f"输出维度: {out_dimension}")
    print(f"输入尺寸: {input_size}")
    
    # 构建模型
    backbone = Backbone(out_dimension=out_dimension, model_name=model_name, pretrained=False, model_path=None)
    model, _, _ = backbone.build_model()
    
    # 加载训练好的权重
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("成功加载模型权重")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return False
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_size)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 导出ONNX模型
        torch.onnx.export(
            model,                          # PyTorch模型
            dummy_input,                    # 示例输入
            output_path,                    # 输出路径
            export_params=True,             # 导出参数
            opset_version=11,              # ONNX算子集版本
            do_constant_folding=True,       # 常量折叠优化
            input_names=['input'],          # 输入节点名称
            output_names=['output'],        # 输出节点名称
            dynamic_axes={
                'input': {0: 'batch_size'},     # 动态批次大小
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX模型已保存至: {output_path}")
        return True
        
    except Exception as e:
        print(f"ONNX转换失败: {e}")
        return False


def verify_onnx_model(onnx_path, input_size=(1, 3, 224, 224)):
    """
    验证ONNX模型的正确性
    
    Args:
        onnx_path (str): ONNX模型路径
        input_size (tuple): 输入张量尺寸
    
    Returns:
        bool: 验证是否成功
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # 加载并验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型结构验证通过")
        
        # 创建推理会话
        ort_session = ort.InferenceSession(onnx_path)
        
        # 创建测试输入
        test_input = torch.randn(input_size).numpy()
        
        # 运行推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"ONNX模型推理成功")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {ort_outputs[0].shape}")
        
        return True
        
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install onnx")
        return False
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为ONNX格式')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='PyTorch模型文件路径(.pth)')
    parser.add_argument('--output_path', type=str, 
                       help='输出ONNX模型路径(.onnx)')
    parser.add_argument('--config', type=str, default='./config/embedding.yaml',
                       help='配置文件路径')
    parser.add_argument('--input_size', type=str, default='1,3,224,224',
                       help='输入张量尺寸，用逗号分隔，如1,3,224,224')
    parser.add_argument('--verify', action='store_true',
                       help='是否验证转换后的ONNX模型')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 设置默认输出路径
    if args.output_path is None:
        base_name = os.path.splitext(os.path.basename(args.model_path))[0]
        args.output_path = f"{base_name}.onnx"
    
    # 解析输入尺寸
    input_size = tuple(map(int, args.input_size.split(',')))
    if len(input_size) != 4:
        print("错误: input_size必须是4个整数，格式为batch,channel,height,width")
        return
    
    # 加载配置文件
    model_config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf8') as f:
                model_config = yaml.safe_load(f)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
    else:
        print(f"配置文件不存在: {args.config}")
        print("使用默认配置")
        model_config = {
            "out_dimension": 256,
            "model_name": "resnet18"
        }
    
    # 执行转换
    success = convert_model_to_onnx(args.model_path, args.output_path, model_config, input_size)
    
    if success and args.verify:
        print("\n正在验证ONNX模型...")
        verify_onnx_model(args.output_path, input_size)


if __name__ == '__main__':
    main()


# python convert_to_onnx.py --model_path /root/autodl-tmp/backup/logs/model-acc-122-0.9559-0.9783-0.9626.pth --output_path /root/autodl-tmp/backup/logs/model-acc-122-0.9559-0.9783-0.9626.onnx --verify