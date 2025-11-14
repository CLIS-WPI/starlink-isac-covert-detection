#!/usr/bin/env python3
"""
🧪 Run Tests with GPU Monitoring
==================================
Executes tests while monitoring GPU utilization in real-time.
"""
import subprocess
import threading
import time
import sys
import os
from pathlib import Path

def monitor_gpu():
    """Monitor GPU utilization in background."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("\n📊 GPU Monitor Started")
        print("="*70)
        
        while True:
            try:
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Format output
                gpu_util = util.gpu
                mem_util = util.memory
                mem_used_gb = memory_info.used / (1024**3)
                mem_total_gb = memory_info.total / (1024**3)
                
                print(f"\r🔥 GPU: {gpu_util}% | Memory: {mem_util}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB) | Temp: {temp}°C", end='', flush=True)
                
                time.sleep(1)
            except Exception as e:
                break
    except ImportError:
        print("⚠️  pynvml not installed. GPU monitoring disabled.")
        print("   Install with: pip install nvidia-ml-py3")
    except Exception as e:
        print(f"⚠️  GPU monitoring failed: {e}")


def check_and_configure_gpu():
    """Check GPU availability and configure TensorFlow."""
    print("="*70)
    print("🔍 Checking GPU Availability...")
    print("="*70)
    
    try:
        import tensorflow as tf
        
        # List all physical devices
        print("\n📋 Available Devices:")
        physical_devices = tf.config.list_physical_devices()
        for device in physical_devices:
            print(f"   - {device}")
        
        # Check GPU specifically
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                
                # Enable memory growth
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"   ✅ Memory growth enabled")
                except Exception as e:
                    print(f"   ⚠️  Could not enable memory growth: {e}")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                result = c.numpy()
                print(f"   ✅ GPU computation successful!")
                print(f"   Result: {result}")
            
            return True
        else:
            print("\n⚠️  No GPU devices found")
            print("   TensorFlow will use CPU")
            return False
            
    except Exception as e:
        print(f"\n❌ Error checking GPU: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Run tests with GPU monitoring."""
    print("\n" + "="*70)
    print("🧪 Running Tests with GPU Monitoring")
    print("="*70)
    
    # Check and configure GPU
    has_gpu = check_and_configure_gpu()
    
    # Start GPU monitoring in background (if GPU available)
    monitor_thread = None
    if has_gpu:
        try:
            import pynvml
            monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
            monitor_thread.start()
        except:
            pass
    
    print("\n" + "="*70)
    print("🚀 Starting Test Execution...")
    print("="*70)
    print()
    
    # Run pytest
    result = subprocess.run(
        [
            'python3', '-m', 'pytest',
            'tests/',
            '-v',
            '--json-report',
            '--json-report-file=test_results.json',
            '--tb=short',
            '--maxfail=10',
        ],
        text=True
    )
    
    print("\n" + "="*70)
    print("✅ Test Execution Complete")
    print("="*70)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(run_tests())

