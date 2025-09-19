#!/usr/bin/env python3
"""
Script to verify and provide information about the converted ONNX models.
"""

import os
from pathlib import Path
import onnx

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def analyze_onnx_model(onnx_path):
    """Analyze ONNX model and return key information."""
    try:
        model = onnx.load(str(onnx_path))
        
        # Get input and output information
        inputs = []
        for input_tensor in model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            inputs.append({
                'name': input_tensor.name,
                'shape': shape,
                'dtype': input_tensor.type.tensor_type.elem_type
            })
        
        outputs = []
        for output_tensor in model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            outputs.append({
                'name': output_tensor.name,
                'shape': shape,
                'dtype': output_tensor.type.tensor_type.elem_type
            })
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'opset_version': model.opset_import[0].version if model.opset_import else 'Unknown'
        }
    except Exception as e:
        return {'error': str(e)}

def verify_onnx_exports():
    """Verify all ONNX exports and provide detailed information."""
    
    checkpoints_dir = Path("checkpoints")
    
    print("🔍 ONNX Model Verification and Analysis")
    print("=" * 60)
    
    # Find all ONNX files
    onnx_files = list(checkpoints_dir.glob("*.onnx"))
    
    if not onnx_files:
        print("❌ No ONNX files found!")
        return False
    
    print(f"Found {len(onnx_files)} ONNX models:\n")
    
    for onnx_file in sorted(onnx_files):
        print(f"📁 {onnx_file.name}")
        print(f"   Size: {get_file_size_mb(onnx_file):.1f} MB")
        
        # Analyze the model
        analysis = analyze_onnx_model(onnx_file)
        
        if 'error' in analysis:
            print(f"   ❌ Error analyzing model: {analysis['error']}")
        else:
            print(f"   ONNX Opset: {analysis['opset_version']}")
            
            # Input information
            if analysis['inputs']:
                input_info = analysis['inputs'][0]  # Usually just one input for YOLO
                print(f"   Input: {input_info['name']} {input_info['shape']}")
            
            # Output information
            if analysis['outputs']:
                for i, output_info in enumerate(analysis['outputs']):
                    print(f"   Output {i+1}: {output_info['name']} {output_info['shape']}")
        
        print()
    
    # Comparison table
    print("📊 Model Comparison for Raspberry Pi:")
    print("-" * 80)
    print(f"{'Model':<15} {'Size (MB)':<12} {'Parameters':<12} {'Best For':<30}")
    print("-" * 80)
    
    model_info = [
        ("yolo11n.onnx", "Fastest inference, lowest memory"),
        ("yolo11s.onnx", "Balance of speed and accuracy"),
        ("yolov10n.onnx", "Latest architecture, efficient"),
        ("yolov10s.onnx", "Good accuracy, moderate speed"),
        ("yolov8n.onnx", "Proven performance, good speed"),
        ("yolov9s.onnx", "High accuracy, slower inference"),
    ]
    
    for model_name, description in model_info:
        model_path = checkpoints_dir / model_name
        if model_path.exists():
            size = get_file_size_mb(model_path)
            print(f"{model_name:<15} {size:<12.1f} {'~2-9M':<12} {description:<30}")
    
    print("-" * 80)
    print("\n🚀 Recommendations for Raspberry Pi:")
    print("1. yolo11n.onnx - Best choice for real-time applications")
    print("2. yolov10n.onnx - Latest architecture with good efficiency")
    print("3. yolov8n.onnx - Reliable option with proven performance")
    print("\n💡 Usage with ONNX Runtime:")
    print("   pip install onnxruntime")
    print("   # For inference:")
    print("   yolo predict model=checkpoints/yolo11n.onnx source=image.jpg")
    
    return True

if __name__ == "__main__":
    verify_onnx_exports()