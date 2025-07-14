# 测试多个随机种子
seeds = [1, 42, 123, 456, 789, 999, 1234, 5678]
results = []

for seed in seeds:
    print(f"🎲 测试种子: {seed}")
    # 设置种子并运行
    result = run_with_seed(seed)  # 需要实现这个函数
    results.append((seed, result))
    print(f"   ARI: {result['ari']:.4f}, NMI: {result['nmi']:.4f}")

# 找最佳种子
best_result = max(results, key=lambda x: x[1]['ari'])
print(f"\n🏆 最佳种子: {best_result[0]}, ARI: {best_result[1]['ari']:.4f}")
