# SearchProtein
    该项目是基于蛋白质分子小角散射信号的蛋白质分子复合物结构优化方法，由特定蛋白质分子复合物在溶液中的唯一小角散射信号来搜索目标蛋白质分子复合物。
    分为两个版本，no_chimera与with_chimera两个版本，前者可在任何python3的环境下运行，不能实时展示移动过程中的效果，速度比较快，后者需要在UCSF Chimera软件中运行，可以实时展示效果，速度比较慢。
    打分函数使用的是当前复合物的信号值与目标复合物的实验数据之间的差异，使用欧氏距离进行度量，在计算过程中，当前复合物的信号值与实验数据之间具有不同大小的维度，需要给实验数据做内插。
    优化算法使用的是组合优化中的模拟退火算法。
    在平移与旋转的过程中，可能会使得两个分子之间的距离变得非常小或者非常大，前者导致两个分子重叠，后者导致两个分子之间失去作用力，在平移与旋转的过程中需要控制这一情况的发生，即需要控制两个分子之间的距离。当两个分子之间的距离非常近时需要将其拉开一定的距离，否则需要将其拉近距离。在程序实现中使用的是计算两个分子中各个原子之间的距离，选取最小距离进行判断。
    对于搜索过程有比较大影响的是初始化以及旋转平移的方向，旋转平移的方向使用的是随机的方式，初始位置的选取是根据高斯分布在固定分子的球面上选取一定数量的采样点，并计算这些采样点与目标复合物之间的差异，选取最小的作为初始位置。
