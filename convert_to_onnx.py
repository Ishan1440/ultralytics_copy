#!/usr/bin/env python3
"""
Script to convert all YOLO checkpoint files to ONNX format for Raspberry Pi deployment.
Optimized for edge devices with appropriate settings.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def convert_checkpoints_to_onnx():
    """Convert all checkpoint files in the checkpoints folder to ONNX format."""
    
    # Define paths
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return False
    
    # Find all .pt files in checkpoints directory
    checkpoint_files = list(checkpoints_dir.glob("*.pt"))
    
    if not checkpoint_files:
        print(f"❌ No .pt checkpoint files found in {checkpoints_dir}")
        return False
    
    print(f"🔍 Found {len(checkpoint_files)} checkpoint files:")
    for file in checkpoint_files:
        print(f"   - {file.name}")
    
    print("\n🚀 Starting ONNX conversion for Raspberry Pi deployment...")
    
    successful_conversions = []
    failed_conversions = []
    
    for checkpoint_file in checkpoint_files:
        try:
            print(f"\n📦 Converting {checkpoint_file.name} to ONNX...")
            
            # Load the model
            model = YOLO(str(checkpoint_file))
            
            # Export to ONNX with Raspberry Pi optimized settings
            onnx_path = model.export(
                format="onnx",
                imgsz=640,  # Standard input size
                optimize=True,  # Optimize for inference
                half=False,  # Use FP32 for better compatibility on CPU
                dynamic=False,  # Static input shape for better performance
                simplify=True,  # Simplify the model
                opset=11,  # ONNX opset version for better compatibility
            )
            
            print(f"✅ Successfully converted {checkpoint_file.name}")
            print(f"   ONNX file: {onnx_path}")
            successful_conversions.append((checkpoint_file.name, onnx_path))
            
        except Exception as e:
            print(f"❌ Failed to convert {checkpoint_file.name}: {str(e)}")
            failed_conversions.append((checkpoint_file.name, str(e)))
    
    # Summary
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successful: {len(successful_conversions)}")
    print(f"❌ Failed: {len(failed_conversions)}")
    
    if successful_conversions:
        print(f"\n✅ Successfully converted models:")
        for original, onnx_path in successful_conversions:
            # Get file size for reference
            if os.path.exists(onnx_path):
                size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"   - {original} → {Path(onnx_path).name} ({size_mb:.1f} MB)")
            else:
                print(f"   - {original} → {Path(onnx_path).name}")
    
    if failed_conversions:
        print(f"\n❌ Failed conversions:")
        for original, error in failed_conversions:
            print(f"   - {original}: {error}")
    
    return len(failed_conversions) == 0

if __name__ == "__main__":
    success = convert_checkpoints_to_onnx()
    sys.exit(0 if success else 1)