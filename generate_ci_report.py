#!/usr/bin/env python3
"""
CI Report Generator for Scenario B EQ Pipeline
Generates median/IQR for preservation and % samples ≥0.5
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_dataset(dataset_path):
    """Load dataset and extract EQ metrics."""
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    meta_list = dataset.get('meta', [])
    
    preservations = []
    snr_improvements = []
    alpha_ratios = []
    flag_fails = []
    
    for meta in meta_list:
        if isinstance(meta, tuple):
            _, meta = meta
        
        if 'eq_pattern_preservation' in meta:
            preservations.append(meta.get('eq_pattern_preservation', 0))
        if 'eq_snr_improvement_db' in meta:
            snr_improvements.append(meta.get('eq_snr_improvement_db', 0))
        if 'alpha_ratio' in meta:
            alpha_ratios.append(meta.get('alpha_ratio', 0))
        if 'flag_csi_fail' in meta:
            flag_fails.append(meta.get('flag_csi_fail', 0))
    
    return {
        'preservations': np.array(preservations),
        'snr_improvements': np.array(snr_improvements),
        'alpha_ratios': np.array(alpha_ratios),
        'flag_fails': np.array(flag_fails),
        'n_samples': len(meta_list)
    }

def generate_ci_report(dataset_paths, output_file='ci_report.txt'):
    """Generate CI report from multiple dataset runs."""
    
    results = []
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        if not Path(dataset_path).exists():
            print(f"⚠️  Run {i}: Dataset not found: {dataset_path}")
            continue
        
        data = load_dataset(dataset_path)
        pres = data['preservations']
        snr_imp = data['snr_improvements']
        alpha_rat = data['alpha_ratios']
        flag_fail = data['flag_fails']
        
        # Compute statistics
        pres_median = np.median(pres)
        pres_q25 = np.percentile(pres, 25)
        pres_q75 = np.percentile(pres, 75)
        pres_iqr = pres_q75 - pres_q25
        
        pres_ge_05 = np.sum(pres >= 0.5)
        pres_ge_05_pct = 100 * pres_ge_05 / len(pres)
        
        snr_mean = np.mean(snr_imp)
        snr_median = np.median(snr_imp)
        
        alpha_valid = alpha_rat[alpha_rat > 0]
        alpha_in_range = np.sum((alpha_valid >= 0.1) & (alpha_valid <= 3.0))
        alpha_in_range_pct = 100 * alpha_in_range / len(alpha_valid) if len(alpha_valid) > 0 else 0
        
        csi_fail_pct = 100 * np.sum(flag_fail) / len(flag_fail)
        
        results.append({
            'run': i,
            'n_samples': data['n_samples'],
            'pres_median': pres_median,
            'pres_q25': pres_q25,
            'pres_q75': pres_q75,
            'pres_iqr': pres_iqr,
            'pres_ge_05': pres_ge_05,
            'pres_ge_05_pct': pres_ge_05_pct,
            'snr_mean': snr_mean,
            'snr_median': snr_median,
            'alpha_in_range_pct': alpha_in_range_pct,
            'csi_fail_pct': csi_fail_pct
        })
    
    if not results:
        print("❌ No valid datasets found!")
        return
    
    # Aggregate statistics
    df = pd.DataFrame(results)
    
    # Generate report
    report = []
    report.append("="*80)
    report.append("📊 CI REPORT: Scenario B EQ Pipeline")
    report.append("="*80)
    report.append("")
    
    report.append("📈 Pattern Preservation Statistics:")
    report.append("-"*80)
    for r in results:
        report.append(f"Run {r['run']}: median={r['pres_median']:.3f}, "
                     f"IQR=[{r['pres_q25']:.3f}, {r['pres_q75']:.3f}], "
                     f"≥0.5: {r['pres_ge_05']}/{r['n_samples']} ({r['pres_ge_05_pct']:.1f}%)")
    
    report.append("")
    report.append("📊 Aggregated Statistics:")
    report.append("-"*80)
    report.append(f"Median preservation: {df['pres_median'].mean():.3f} "
                 f"(std: {df['pres_median'].std():.3f})")
    report.append(f"Mean IQR: {df['pres_iqr'].mean():.3f}")
    report.append(f"Mean % ≥0.5: {df['pres_ge_05_pct'].mean():.1f}% "
                 f"(std: {df['pres_ge_05_pct'].std():.1f}%)")
    report.append(f"Target range: 38-42%")
    report.append("")
    
    report.append("📊 SNR Improvement:")
    report.append("-"*80)
    for r in results:
        report.append(f"Run {r['run']}: mean={r['snr_mean']:.2f} dB, "
                     f"median={r['snr_median']:.2f} dB")
    report.append(f"Overall mean: {df['snr_mean'].mean():.2f} dB")
    report.append("")
    
    report.append("📊 Alpha Ratio:")
    report.append("-"*80)
    for r in results:
        report.append(f"Run {r['run']}: {r['alpha_in_range_pct']:.1f}% in range 0.1x-3x")
    report.append(f"Overall mean: {df['alpha_in_range_pct'].mean():.1f}%")
    report.append("")
    
    report.append("📊 CSI Failures:")
    report.append("-"*80)
    for r in results:
        report.append(f"Run {r['run']}: {r['csi_fail_pct']:.1f}%")
    report.append("")
    
    report.append("="*80)
    report.append("✅ Acceptance Criteria:")
    report.append("="*80)
    pres_ok = df['pres_median'].mean() >= 0.48  # Close to 0.5
    snr_ok = df['snr_mean'].mean() >= 4.0
    alpha_ok = df['alpha_in_range_pct'].mean() >= 80.0
    csi_ok = df['csi_fail_pct'].mean() < 5.0
    
    report.append(f"Pattern Preservation (median ≥0.48): "
                 f"{'✅ PASS' if pres_ok else '❌ FAIL'} "
                 f"({df['pres_median'].mean():.3f})")
    report.append(f"SNR Improvement (≥4 dB): "
                 f"{'✅ PASS' if snr_ok else '❌ FAIL'} "
                 f"({df['snr_mean'].mean():.2f} dB)")
    report.append(f"Alpha Ratio (≥80% in range): "
                 f"{'✅ PASS' if alpha_ok else '❌ FAIL'} "
                 f"({df['alpha_in_range_pct'].mean():.1f}%)")
    report.append(f"CSI Failures (<5%): "
                 f"{'✅ PASS' if csi_ok else '❌ FAIL'} "
                 f"({df['csi_fail_pct'].mean():.1f}%)")
    report.append("="*80)
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Report saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    import sys
    
    # Default dataset paths
    dataset_paths = [
        'dataset/dataset_scenario_b_500.pkl',
        'dataset/dataset_scenario_b_500_run2.pkl',
        'dataset/dataset_scenario_b_500_run3.pkl'
    ]
    
    if len(sys.argv) > 1:
        dataset_paths = sys.argv[1:]
    
    generate_ci_report(dataset_paths, 'ci_report.txt')

