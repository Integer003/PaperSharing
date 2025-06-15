---
title: HuB
published: 2025-05-30
description: 'Learning Extreme Humanoid Balance'
image: ''
tags: [TXC, IL]
category: 'Paper'
draft: false 
lang: 'zh_CN'
---

# HuB: Learning Extreme Humanoid Balance

Motivation：人类可以展示出很好的平衡能力，但机器人多数容易倒。

问题：人形机器人能够有极端平衡性吗？如何获得控制策略让机器人有这个平衡能力？

之前的让人形机器人跟踪人类动作的pipeline：OmniH2O，跟踪人类行动，关键是获得人类行动的数据，这可以通过两种方法实现：

1. 视频得到MoCap（动作捕捉）
2. 以标记器为基础的MoCap系统

然后从获得的人类行动retarget到人形机器人身上。这又分为两步实现：

1. 在仿真环境中训练一个control policy来跟踪参考motion（通常用teacher-student学习策略）
2. 部署到实际机器人身上

但是以上pipeline在应用到极端平衡问题中通常有一些困难：

1. 参考motion需要有高精确度，不能有大error
    MoCap中引入的error：

    + video-based会有很大的误差
    + marker-based虽然精度高，但是Internet中的多数视频都没有marker，很难有消费级数据集

    Retargeting引入的error：

    + 优化问题非凸
    + model alignment不完美
    + 缺乏时序连续性限制

2. 训练Balance Policies的困难：

    + 形态差异
    + 质心不一定对齐
    + 因此，绝对服从人类动作数据不能保证机器人有稳定的平衡

3. Sim2Real Transfer

    + 传感器噪声大（IMUs，VIO systems）
    + 以前的方法在平衡问题中，摇晃和抖动很严重

针对以上三个问题，论文提出了对应的三种方法来解决。

1. 为了解决Reference Motion Error带来的挑战，我们对数据集中的Reference Motion进行Refine。

    + SMPL-Initialized Retargeting
        Details：之前的方法都是先把人形机器人的关节初始化为0，然后开始优化，但对于非凸优化问题，这个初始化可能离最优解非常远，优化很困难。因此，我们使用SMPL-Initialized方法，由于SMPL数据库中包含了人形机器人g1的29个dof的所有信息，所以我们根据给定的姿势选取SMPL中类似的姿势然后取其最优

    + Grounded-Foot Correction
        由于Retargeting引入的error大，经常会发现人类运动数据集的单脚站立场景中这个脚相对地面会滑动，这显然不符合物理。为了消除这种相对地面的滑动，论文使用了很简单的方法：保持所有关节角度不变，移动不同帧中root_link的position，让所有帧的接地脚位置不动。经过这一步数据预处理后，可以想见policy就应该会学的更好。

    + Center-of-Mass Filtering
        如果数据集中的某个trajectories在某时发现用urdf文件、当前dof角度计算出的质心位置离支撑脚的重心超过0.2m，那么整个trajectories直接丢弃。

    + Transition Stablization

        > double support phase 指的是在步态周期（gait cycle）中，两个脚都与地面接触的时间段。
        >
        > 具体来说，在人类行走或机器狗/双足机器人的行走中：
        >
        > - 一步完整的行走动作包括 **摆动相（swing phase）** 和 **支撑相（stance phase）**。
        > - 当左脚刚迈出，而右脚还没完全抬起时，两只脚都在地面上，这段时间就是 **double support phase**。
        > - 对于 **跑步或奔跑（running）** 来说，通常**没有** double support phase，取而代之的是**flight phase（腾空相）**。

        为了能提升稳定性，在目标高难度平衡动作前后添加更多double support phase动作。

2. 为了训练Balance Policies，引入“Balance-Aware Policy Learning”, 即在policy训练过程中有意让人形机器人更加平衡

    + Relaxed Reference Tracking
        由于人类和人形机器人物理属性上本就不同，以及考虑到MoCap过程中引入的误差，严格跟踪动作捕捉得到的关键点的话机器人不太能平衡。因此允许适度偏离动作捕捉得到的关键点，这样可以大大提升人形机器人的稳定性。
    + Balance shape reward
        仅有上一点不足以让机器人稳定，因此还需要引入提升机器人稳定性的多种heuristic(人类对如何保持平衡的先验知识)
        + Center-of-Mass reward: 奖励重心能够位于脚掌内的姿势
        + Foot contact mismatch penalty: 如果参考motion中有一只脚离地，那么跟踪的motion也应该有相同的脚离地（估计因为policy为了避免不平衡的惩罚，最终选择尽管参考运动中有一只脚离地，我也为保持平衡选择双脚都不离地）
        + Close feet penalty: 双脚太近说明稳定性比较差

3. 为了让sim2real足够robust，论文采用了以下几种方法：

    + Localized Reference Tracking
        在student 训练和部署的时候不使用里程计（即空间绝对位置），所有tracking target都在人形机器人局部坐标系中表示
    + IMU-Centric Perturbation
        对观测到的root link的方向执行OU noise的扰动，之后计算其他观测值的时候也基于这个扰动后的root link的方向计算
    + High-Frequency Push
        训练中执行高频但较小的扰动，例如每1秒给0.5m/s的速度。之前方法用的是低频但很大的扰动，由于我们的平衡问题允许的身体变化范围很小，使用低频大扰动不太可能还能平衡，所以我们改用高频小扰动。