#!/usr/bin/env python3
"""
Measure CNN inference performance for real-time detection analysis.
Usage: python3 measure_inference_performance.py [model_path]
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

def measure_inference_performance(model_path, scenario='a'):
    """Measure inference time and throughput."""
    
    print(f"\n{'='*70}")
    print(f"Measuring Inference Performance: {model_path}")
    print(f"{'='*70}")
    
    # Load model
    print(f"\n📁 Loading model...")
    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get input shape from model
    input_shape = model.input_shape[1:]  # Remove batch dimension
    print(f"   Input shape: {input_shape}")
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Model size
    model_size_bytes = Path(model_path).stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"   Model file size: {model_size_mb:.2f} MB")
    
    # Create test samples
    print(f"\n🔧 Creating test samples...")
    single_sample = np.random.randn(1, *input_shape).astype(np.float32)
    batch_8 = np.random.randn(8, *input_shape).astype(np.float32)
    batch_32 = np.random.randn(32, *input_shape).astype(np.float32)
    batch_64 = np.random.randn(64, *input_shape).astype(np.float32)
    
    # Warm-up (first inference is slower due to graph compilation)
    print(f"   Warming up (3 runs)...")
    for _ in range(3):
        _ = model.predict(single_sample, verbose=0)
        _ = model.predict(batch_8, verbose=0)
        _ = model.predict(batch_32, verbose=0)
        _ = model.predict(batch_64, verbose=0)
    
    # Measure single sample
    print(f"\n⏱️  Measuring single-sample inference (1000 runs)...")
    n_runs = 1000
    times_single = []
    
    for i in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(single_sample, verbose=0)
        end = time.perf_counter()
        times_single.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i+1}/{n_runs} runs...")
    
    mean_single = np.mean(times_single)
    std_single = np.std(times_single)
    p50_single = np.percentile(times_single, 50)
    p95_single = np.percentile(times_single, 95)
    p99_single = np.percentile(times_single, 99)
    
    print(f"\n   📊 Single-Sample Results:")
    print(f"      Mean:     {mean_single:.3f} ± {std_single:.3f} ms")
    print(f"      Median:   {p50_single:.3f} ms")
    print(f"      P95:      {p95_single:.3f} ms")
    print(f"      P99:      {p99_single:.3f} ms")
    print(f"      Throughput: {1000/mean_single:.1f} samples/sec")
    
    # Measure batch 8
    print(f"\n⏱️  Measuring batch-8 inference (200 runs)...")
    times_batch8 = []
    
    for i in range(200):
        start = time.perf_counter()
        _ = model.predict(batch_8, verbose=0)
        end = time.perf_counter()
        times_batch8.append((end - start) * 1000)  # ms
    
    mean_batch8_total = np.mean(times_batch8)
    mean_batch8_per_sample = mean_batch8_total / 8
    std_batch8_per_sample = np.std(times_batch8) / 8
    throughput_batch8 = 8 * 200 / (np.sum(times_batch8) / 1000)  # samples/sec
    
    print(f"\n   📊 Batch-8 Results:")
    print(f"      Total batch time: {mean_batch8_total:.3f} ms")
    print(f"      Per sample:       {mean_batch8_per_sample:.3f} ± {std_batch8_per_sample:.3f} ms")
    print(f"      Throughput:       {throughput_batch8:.1f} samples/sec")
    
    # Measure batch 32
    print(f"\n⏱️  Measuring batch-32 inference (200 runs)...")
    times_batch32 = []
    
    for i in range(200):
        start = time.perf_counter()
        _ = model.predict(batch_32, verbose=0)
        end = time.perf_counter()
        times_batch32.append((end - start) * 1000)  # ms
    
    mean_batch32_total = np.mean(times_batch32)
    mean_batch32_per_sample = mean_batch32_total / 32
    std_batch32_per_sample = np.std(times_batch32) / 32
    throughput_batch32 = 32 * 200 / (np.sum(times_batch32) / 1000)  # samples/sec
    
    print(f"\n   📊 Batch-32 Results:")
    print(f"      Total batch time: {mean_batch32_total:.3f} ms")
    print(f"      Per sample:       {mean_batch32_per_sample:.3f} ± {std_batch32_per_sample:.3f} ms")
    print(f"      Throughput:       {throughput_batch32:.1f} samples/sec")
    
    # Measure batch 64
    print(f"\n⏱️  Measuring batch-64 inference (200 runs)...")
    times_batch64 = []
    
    for i in range(200):
        start = time.perf_counter()
        _ = model.predict(batch_64, verbose=0)
        end = time.perf_counter()
        times_batch64.append((end - start) * 1000)  # ms
    
    mean_batch64_total = np.mean(times_batch64)
    mean_batch64_per_sample = mean_batch64_total / 64
    std_batch64_per_sample = np.std(times_batch64) / 64
    throughput_batch64 = 64 * 200 / (np.sum(times_batch64) / 1000)  # samples/sec
    
    print(f"\n   📊 Batch-64 Results:")
    print(f"      Total batch time: {mean_batch64_total:.3f} ms")
    print(f"      Per sample:       {mean_batch64_per_sample:.3f} ± {std_batch64_per_sample:.3f} ms")
    print(f"      Throughput:       {throughput_batch64:.1f} samples/sec")
    
    # Real-time feasibility analysis
    print(f"\n{'='*70}")
    print("🎯 Real-Time Feasibility Analysis:")
    print(f"{'='*70}")
    
    ofdm_frame_duration = 10.0  # ms (10 symbols × 1 ms per symbol)
    print(f"\nOFDM Frame Duration: {ofdm_frame_duration} ms")
    print(f"Detection Deadline: < {ofdm_frame_duration} ms per frame")
    
    print(f"\n📊 Single-Sample:")
    if mean_single < ofdm_frame_duration:
        margin = ofdm_frame_duration / mean_single
        print(f"   ✅ REAL-TIME FEASIBLE")
        print(f"   Latency: {mean_single:.3f} ms ({margin:.1f}x faster than required)")
        print(f"   P99 latency: {p99_single:.3f} ms (still feasible)")
    else:
        speedup_needed = mean_single / ofdm_frame_duration
        print(f"   ❌ NOT REAL-TIME")
        print(f"   Latency: {mean_single:.3f} ms ({speedup_needed:.1f}x too slow)")
        print(f"   Need optimization or GPU acceleration")
    
    print(f"\n📊 Batch-8:")
    if mean_batch8_per_sample < ofdm_frame_duration:
        margin = ofdm_frame_duration / mean_batch8_per_sample
        print(f"   ✅ REAL-TIME FEASIBLE")
        print(f"   Latency: {mean_batch8_per_sample:.3f} ms/sample ({margin:.1f}x faster)")
    else:
        print(f"   ⚠️  MARGINAL")
        print(f"   Latency: {mean_batch8_per_sample:.3f} ms/sample")
    
    print(f"\n📊 Batch-32:")
    if mean_batch32_per_sample < ofdm_frame_duration:
        margin = ofdm_frame_duration / mean_batch32_per_sample
        print(f"   ✅ REAL-TIME FEASIBLE (OPTIMAL)")
        print(f"   Latency: {mean_batch32_per_sample:.3f} ms/sample ({margin:.1f}x faster)")
    else:
        print(f"   ⚠️  MARGINAL")
        print(f"   Latency: {mean_batch32_per_sample:.3f} ms/sample")
    
    print(f"\n📊 Batch-64:")
    if mean_batch64_per_sample < ofdm_frame_duration:
        margin = ofdm_frame_duration / mean_batch64_per_sample
        print(f"   ✅ REAL-TIME FEASIBLE (OPTIMAL)")
        print(f"   Latency: {mean_batch64_per_sample:.3f} ms/sample ({margin:.1f}x faster)")
    else:
        print(f"   ⚠️  MARGINAL")
        print(f"   Latency: {mean_batch64_per_sample:.3f} ms/sample")
    
    # Memory analysis
    print(f"\n{'='*70}")
    print("💾 Memory Analysis:")
    print(f"{'='*70}")
    
    input_size_kb = np.prod(input_shape) * 4 / 1024  # 4 bytes per float32
    print(f"   Input size per sample: {input_size_kb:.2f} KB")
    print(f"   Batch-32 memory: {input_size_kb * 32:.2f} KB")
    print(f"   Batch-64 memory: {input_size_kb * 64:.2f} KB")
    print(f"   Model size: {model_size_mb:.2f} MB")
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 Summary:")
    print(f"{'='*70}")
    print(f"   Best single-sample latency: {mean_single:.3f} ms")
    print(f"   Best batch latency: {min(mean_batch8_per_sample, mean_batch32_per_sample, mean_batch64_per_sample):.3f} ms/sample")
    print(f"   Maximum throughput: {max(1000/mean_single, throughput_batch8, throughput_batch32, throughput_batch64):.0f} samples/sec")
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Parameters: {total_params:,}")
    
    return {
        'scenario': scenario,
        'input_shape': input_shape,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'single_ms': mean_single,
        'single_std': std_single,
        'single_p99': p99_single,
        'batch8_ms': mean_batch8_per_sample,
        'batch32_ms': mean_batch32_per_sample,
        'batch64_ms': mean_batch64_per_sample,
        'throughput_max': max(1000/mean_single, throughput_batch8, throughput_batch32, throughput_batch64),
        'realtime_feasible': mean_single < ofdm_frame_duration
    }

if __name__ == "__main__":
    # Check if model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        measure_inference_performance(model_path)
    else:
        # Default: measure both scenarios
        model_path_a = 'model/scenario_a/cnn_detector.keras'
        model_path_b = 'model/scenario_b/cnn_detector.keras'
        
        results = {}
        
        if Path(model_path_a).exists():
            results['a'] = measure_inference_performance(model_path_a, 'a')
        else:
            print(f"⚠️  Model not found: {model_path_a}")
        
        if Path(model_path_b).exists():
            results['b'] = measure_inference_performance(model_path_b, 'b')
        else:
            print(f"⚠️  Model not found: {model_path_b}")
        
        # Save results
        if results:
            import json
            output_path = 'result/inference_performance.json'
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to native Python types
            results_serializable = {}
            for key, value in results.items():
                if value is not None:
                    results_serializable[key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                        if k != 'input_shape'  # Skip numpy array
                    }
                    results_serializable[key]['input_shape'] = list(value['input_shape'])
            
            with open(output_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            print(f"\n✅ Results saved to: {output_path}")

