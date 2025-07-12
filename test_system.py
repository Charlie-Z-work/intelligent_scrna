#!/usr/bin/env python3
"""快速测试脚本"""

import subprocess
import time

def test_usoskin():
    """测试Usoskin数据集"""
    
    print("🧪 开始Usoskin数据测试...")
    start_time = time.time()
    
    # 运行优化后的系统
    result = subprocess.run([
        "python", "main.py", 
        "--data", "data/in_X.csv", 
        "--labels", "data/true_labs.csv", 
        "--name", "Usoskin_Optimized",
        "--max-iter", "3"
    ], capture_output=True, text=True)
    
    duration = time.time() - start_time
    
    print(f"测试完成，耗时: {duration:.1f}秒")
    
    if result.returncode == 0:
        print("✅ 测试成功!")
        print("输出摘要:")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NMI:' in line or '最终性能:' in line or '总体改进:' in line:
                print(f"  {line.strip()}")
    else:
        print("❌ 测试失败!")
        print("错误信息:")
        print(result.stderr)

if __name__ == "__main__":
    test_usoskin()
