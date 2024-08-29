# 光伏功率预测

fork https://github.com/irutheu/LSTM-power-forecasting.git 仓库。
仓库中使用LSTM模型进行预测。

## 要求

研制基于深度学习的光伏功率预测工具，满足：

- 具备短期光伏功率预测功能；
- 能够基于历史和当前气象数据动态调整预测结果；
- 能够针对不同地区和类型的光伏电站进行模型微调，显著提高实际应用中的预测精度与可靠性。

## 数据说明

| 数据名称             | 输入/标签 | 说明 | 与dc_power的Pearson相关系数 |
| :-----------------: | :------: | :-: | :-----------------------: |
| ac_current          |  |  | - |
| ac_power            |  |  | - |
| ac_voltage          |  |  | - |
| ambient_temp        | 输入 | 环境温度 | 0.415954619 |
| dc_current          |  |  | - |
| dc_power            | 标签 | 直流功率 | 1 |
| dc_voltage          |  |  | - |
| inverter_error_code |  |  | - |
| inverter_temp       | 输入 | 逆变器温度 | 0.633105788 |
| module_temp         | 输入 | 模块温度 | 0.721930821 |
| poa_irradiance      | 输入 | 阵列面辐照度 | 0.980303532 |
| power_factor        |  |  | - |
| relative_humidity   | 输入 | 相对湿度 | -0.405921067 |
| wind_direction      | 输入 | 风向 | -0.027344522 |
| wind_speed          | 输入 | 风速 | 0.214515646 |

## 实验结果

使用 2012-2014 年 hourly 的光伏功率数据，训练 20 个 epochs 后的结果。

| 序号 | 模型             | optimizer | 训练效果 | rmse | mae | 预测曲线图 |
| :-: | :--------------: | :-------: | :-----: | :-: | :-: | :-------: |
| baseline | ![model_baseline](./assets/model_baseline.png) | Adam | 7760534.0645 | 3646 | 2060 | ![figure_baseline](./assets/figure_baseline.png) |
| 1 | ![model_1](./assets/model_1.png) | AdamW | 5225995.5000 | 3148.1853 | 1795.4615 | ![figure_1](./assets/figure_1.png) |
| 2 | ![model_2](./assets/model_2.png) | AdamW | ![history_2](./assets/history_2.png) | 4082.8988 | 3409.0554 | ![figure_2](./assets/figure_2.png) |
| 3 | ![model_3](./assets/model_3.png) | AdamW | ![history_3](./assets/history_3.png) | 3370.2672 | 1834.6124 | ![figure_3](./assets/figure_3.png) |
| 4 | ![model_4](./assets/model_4.png) | AdamW | ![history_4](./assets/history_4.png) | 3598.4814 | 1883.6083 | ![figure_4](./assets/figure_4.png) |
| 5 | ![model_5](./assets/model_5.png) | AdamW | ![history_5](./assets/history_5.png) | 3389.9635 | 1824.6341 | ![figure_5](./assets/figure_5.png) |
| 6 | ![model_6](./assets/model_6.png) | AdamW | ![history_6](./assets/history_6.png) | 3088.3614 | 1506.0205 | ![figure_6](./assets/figure_6.png) |
| 7 | ![model_7](./assets/model_7.png) | AdamW | ![history_7](./assets/history_7.png) | 3227.9855 | 1808.1091 | ![figure_7](./assets/figure_7.png) |
