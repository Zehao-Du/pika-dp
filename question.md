# 开发日志

## 7/06 问题

1. 模型输出和实际控制不在同一时刻,这个特别重要,导致gripper无法抓住小方块.

## 7/07 

1. 更正7.06问题，与控制时刻/延迟无关 
2. 增加 gripper width threshood, 当模型输出小于 threshood时硬编码为0.01, 经验证可以抓起小方块

问题

1. threshood带来，在抓起小方块，运送至 drawer 的过程中，模型不会一直输出小于threshood的值，导致小方块在中途掉落
2. threshood方案有自身的问题，如果第一次尝试robot没有抓住小方块，将会直接OOD,因为数据集里没有gripper width=0.01,并且在小方块附近的数据 

解决方案

1. 调整 threshood 取值