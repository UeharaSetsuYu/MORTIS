import pandas as pd
import matplotlib.pyplot as plt


def plot_melancholic_metrics(datasetname):
    # 1. 读取 Excel 文件
    # 沿用了 Lin 设置好的相对路径和数据集名称命名逻辑哦
    try:
        excel_path = './Result_Doc/' + datasetname + '_Clu_Performance.xlsx'
        df = pd.read_excel(excel_path, sheet_name='Result')
    except Exception as e:
        print(f"那个……文件好像读不出来呢，可以再检查一下路径吗？{e}")
        return

    metrics = ['ACC', 'ARI', 'NMI', 'PUR']
    x_data = df.index

    # 2. 字体与基础设置
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 给画布背景蒙上一层极其微弱的冷灰色，不再是刺眼的纯白
    ax.set_facecolor('#F6F7F9')
    fig.patch.set_facecolor('#FFFFFF')

    # 3. 忧郁色系调配 (褪色、降噪、带灰度的冷色调)
    colors = ['#4EB9D3', '#B2E2DC', '#CAEAF2', '#C5CCDB']
    markers = ['o', 's', '^', 'D']

    # 4. 绘制曲线：线条变细，透明度增加，呈现出一种“易碎感”
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.plot(x_data, df[metric],
                    label=metric,
                    color=colors[i],
                    # marker=markers[i],
                    linewidth=1.8,  # 线条更纤细
                    # markersize=7,
                    alpha=0.8,  # 增加透明度，让颜色沉下去
                    markerfacecolor='#F6F7F9',  # 标记内部和背景融为一体，显得空洞
                    markeredgewidth=1.2)

    # 5. 坐标轴与网格设置
    ax.set_xlabel('Index', fontsize=13, color='#333333')
    ax.set_ylabel('Clustering Performance', fontsize=13, color='#333333')

    # 刻度颜色调暗
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#555555')

    # 网格线变成更细碎、更淡的点阵
    ax.grid(True, linestyle=':', color='#B0B8C1', alpha=0.6)

    # 6. 边框处理：去掉顶部和右侧，剩下的边框变细、颜色变淡
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(1.0)

    # 7. 图例设置
    # 图例的文字颜色也带一点灰
    legend = ax.legend(frameon=False, fontsize=11, loc='best')
    for text in legend.get_texts():
        text.set_color('#444444')

    plt.tight_layout()

    # 8. 保存与显示
    # 按照 Lin 的路径要求保存，并在文件名后加了一点小小的标识呢
    plt.savefig('./Figure/' + datasetname + '_Metrics_Result_Melancholy.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('./Figure/' + datasetname + '_Metrics_Result_Melancholy.png', format='png', dpi=300,
                bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':
    datasetname = 'DHA'
    plot_melancholic_metrics(datasetname)
