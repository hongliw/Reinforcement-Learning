4*4方格世界如下图所示：

![节点](.//Users/yeungharvey/Documents/ReinforcementLearning/Reinforcement-Learning/Machine\ Learning\ Book/Chapter\ 16-3\ model-Based/grid.png) 


1、状态空间 S：S1-S14为非终止状态；S0、S15为终止状态；

2、行为空间 A：{n,e,e,w}对于任何非终止状态可以有向北、东、南、西四个行为；

3、转移概率 P：任何试图离开方格世界的动作其位置将不会发生改变；其余条件下将100%地转移到动作指向的位置；

4、即时奖励 R：任何非终止状态间的转移得到的即时奖励均为-1；进入终止状态即时奖励为0；

5、衰减系数 gamma: 1;

6、当前策略 ：个体采用随机行动策略

更详细参考https://zhuanlan.zhihu.com/p/28084990