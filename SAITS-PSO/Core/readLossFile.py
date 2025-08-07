from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
if __name__ == "__main__":
    filename = r'Y:\宋启航\data\小球数据差分\forpaper\y2000\20241208_T145346\tensorboard' \
               r'\events.out.tfevents.1733640826.DigiTOM-WorkStation-C210-3.20052.0.pypots'
    # 加载日志文件
    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()  # 必须先加载

    # 获取所有可用的 tags
    print("Available scalar tags:", ea.Tags()['scalars'])

    # 获取指定 tag 的数据（如标量数据）
    val_loss = ea.Scalars('validating/imputation_loss')  # 替换 'tag_name' 为你感兴趣的标签
    train_loss = ea.Scalars('training/loss')  # 替换 'tag_name' 为你感兴趣的标签
    # # 输出标量数据
    # for item in val_loss:
    #     print(f"Step: {item.step}, Value: {item.value}, Wall Time: {item.wall_time}")
    # 提取数据点
    # 提取原始数据点
    train_steps = np.array([point.step for point in train_loss])
    train_values = np.array([point.value for point in train_loss])
    val_steps = np.array([point.step for point in val_loss])
    val_values = np.array([point.value for point in val_loss])

    # 计算间隔采样步长
    sampling_step = len(train_steps) // len(val_steps)

    # 间隔采样
    train_steps_sampled = train_steps[::sampling_step][:len(val_steps)]
    train_values_sampled = train_values[::sampling_step][:len(val_steps)]

    # 绘制图形
    plt.figure(figsize=(8, 6))
    plt.plot(val_steps, train_values_sampled, label='Training Loss', color='blue', linestyle='-',
             linewidth=2)
    plt.plot(val_steps, val_values, label='Validation Loss', color='orange', linestyle='-', linewidth=2)

    # 图形设置
    plt.xlabel('Epochs', fontsize=24, family='Arial')
    plt.ylabel('Loss', fontsize=24, family='Arial')
    plt.title('Training and Validation Loss', fontsize=24, family='Arial')
    custom_font = fm.FontProperties(family='Arial', size=12)
    plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True, prop=custom_font)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, fontweight='bold', family='Times New Roman')
    plt.yticks(fontsize=12, fontweight='bold', family='Times New Roman')

    # 保存图片（可选）
    plt.savefig('sampled_loss_curve.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()