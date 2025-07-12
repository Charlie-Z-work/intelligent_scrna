# intelligent_scrna

智能单细胞RNA测序分析系统

## 📁 项目结构

intelligent_scrna/
 ├── main.py                    # 主控文件
 ├── config.py                  # 配置文件
 ├── core/
 │   ├── geometry_analyzer.py   # 几何特征分析器
 │   ├── failure_analyzer.py    # 失败模式诊断器
 │   ├── strategy_atlas.py      # 策略图谱管理器
 │   └── iterative_learner.py   # 迭代学习器
 ├── algorithms/
 │   ├── boundary_failure.py    # 边界失败学习算法
 │   ├── enhanced_sre.py        # 增强SRE算法
 │   └── ultimate_fusion.py     # 终极融合算法
 ├── utils/
 │   ├── data_loader.py         # 数据加载工具
 │   ├── metrics.py             # 评估指标
 │   └── visualization.py       # 可视化工具
 ├── data/                      # 数据目录
 ├── results/                   # 输出目录
 ├── logs/                      # 日志目录
 └── cache/                     # 缓存目录

  
