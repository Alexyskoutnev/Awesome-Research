# Pretty Cool Research Papers

Welcome to my curated collection of research papers in the fields of Deep Learning, Reinforcement Learning, Machine Learning, and Robotics. Dive into the latest advancements and gain insights from the forefront of these amazing domains.

## Table of Contents

1. [Deep Learning](#deep-learning)
2. [Reinforcement Learning](#reinforcement-learning)
3. [Multi-Agent Reinforcement Learning](#marl)
4. [Machine Learning](#machine-learning)
5. [Robotics](#robotics)
6. [Game Theory](#game-theory)
7. [Quantum Computing](#quantum-computing)

## Deep Learning

### 1. Attention All You Need
   - **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.
   - **Summary:** This seminal paper introduces the Transformer architecture, which utilizes self-attention mechanisms to capture long-range dependencies in sequences. The Transformer has become a cornerstone in natural language processing and other sequential data tasks, outperforming previous recurrent and convolutional architectures.
   - **Link:** [Attention All You Need](https://arxiv.org/abs/1706.03762)

## Reinforcement Learning

### 1. Playing Atari with Deep Reinforcement Learning
   - **Authors:** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al.
   - **Summary:** This paper introduces Deep Q-Networks (DQN), a groundbreaking approach that combines deep neural networks with Q-learning. The model was trained to play multiple Atari 2600 games, achieving human-level performance.
   - **Link:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

### 2. Proximal Policy Optimization Algorithms
   - **Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, et al.
   - **Summary:** Proximal Policy Optimization (PPO) is introduced as a family of policy optimization algorithms. PPO is known for its stability and ease of implementation, making it widely adopted in both research and practical applications.
   - **Link:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## Multi-Agent Reinforcement Learning 

### 1. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms
   - **Authors:** Kaiqing Zhang, Zhuoran Yang, Tamer Başar
   - **Summary:** This paper provides a comprehensive overview of theories and algorithms in the field of Multi-Agent Reinforcement Learning (MARL). It delves into the challenges and solutions associated with coordinating the learning of multiple agents, offering valuable insights into the evolving landscape of MARL research.
   - **Link:** [Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms](https://arxiv.org/abs/1911.10635)

## Machine Learning

### 1. ImageNet Classification with Deep Convolutional Neural Networks
   - **Authors:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
   - **Summary:** This groundbreaking paper introduces the use of deep convolutional neural networks (CNNs) for image classification tasks. The proposed architecture, known as AlexNet, significantly outperformed existing methods at the time and played a pivotal role in popularizing deep learning for computer vision.
   - **Link:** [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

### 2. A Few Useful Things to Know About Machine Learning
   - **Authors:** Pedro Domingos
   - **Summary:** This paper provides a collection of practical insights and tips for practitioners in the field of machine learning. It covers a range of topics, from data preparation to model evaluation, offering valuable guidance based on the author's extensive experience.
   - **Link:** [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

### 3. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
   - **Authors:** Sergey Ioffe, Christian Szegedy
   - **Summary:** This influential paper introduces Batch Normalization, a technique that significantly accelerates the training of deep neural networks by normalizing intermediate feature activations. It helps mitigate the internal covariate shift problem and has become a standard component in the design of deep learning architectures.
   - **Link:** [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

### 4. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
   - **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
   - **Summary:** This paper introduces the Vision Transformer (ViT) model, which applies the Transformer architecture directly to sequences of image patches for image classification tasks. ViT achieves state-of-the-art results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
   - **Link:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Robotics

### 1. SLAM: Simultaneous Localization and Mapping - Part I
   - **Authors:** Hugh Durrant-Whyte, Tim Bailey
   - **Summary:** This seminal paper provides a comprehensive overview of Simultaneous Localization and Mapping (SLAM) techniques, a crucial aspect of robotic navigation. It covers fundamental concepts and algorithms essential for robots to build maps of their environment while simultaneously determining their own location.
   - **Link:** [SLAM: Simultaneous Localization and Mapping - Part I](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/Durrant-Whyte_Bailey_SLAM-tutorial-I.pdf)

### 2. Learning Torque Control for Quadrupedal Locomotion
   - **Authors:** Shuxiao Chen, Bike Zhang, Mark W. Mueller, Akshara Rai, Koushil Sreenath
   - **Summary:** The paper discusses the shift in paradigm from position-based to torque-based control in reinforcement learning (RL) for quadrupedal robots. It introduces a torque-based RL framework, where the RL policy directly predicts joint torques at a high frequency, eliminating the need for a proportional-derivative (PD) controller. The proposed torque control framework demonstrates superior performance in terms of reward and robustness to external disturbances, marking it as the first sim-to-real attempt for end-to-end learning torque control in quadrupedal locomotion.
   - **Link:** [Learning Torque Control for Quadrupedal Locomotion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10375154)
### 3. Real-World Humanoid Locomotion with Reinforcement Learning
   - **Authors:** Ilija Radosavovic, Tete Xiao, Bike Zhang, Trevor Darrell, Jitendra Malik, Koushil Sreenath
   - **Summary:** This research paper introduces a fully learning-based approach for real-world humanoid locomotion. The proposed controller, a causal transformer, utilizes the history of proprioceptive observations and actions to predict the next action. The model is trained through large-scale model-free reinforcement learning in simulated environments, and it demonstrates the ability to walk over diverse outdoor terrains, adapt in context, and exhibit robustness to external disturbances when deployed in the real world.
   - **Link:**  [Real-World Humanoid Locomotion with Reinforcement Learning](https://arxiv.org/pdf/2303.03381.pdf)
### 4. Learning Bipedal Walking On Planned Footsteps For Humanoid Robots
   - **Authors:** Rohan P. Singh, Mehdi Benallegue, Mitsuharu Morisawa, Rafael Cisneros, Fumio Kanehiro
   - **Summary:** The paper introduces a deep reinforcement learning method that employs a step sequence controller to inform the policy about the future locomotion plan. By training the policy to follow procedurally generated step sequences, the method achieves omnidirectional walking, turning, standing, and stair climbing without the need for reference motions or pre-trained weights. The proposed approach is demonstrated on two new robot platforms, HRP5P and JVRC-1, using the MuJoCo simulation environment.
   - **Link:**  [Learning Bipedal Walking On Planned Footsteps For Humanoid Robots](https://arxiv.org/pdf/2207.12644.pdf)
### 5. Expressive Whole-Body Control for Humanoid Robots
   - **Authors:** Xuxin Cheng , Yandong Ji , Junming Chen, Ruihan Yang, Ge Yang, Xiaolong Wang
   - **Summary:** This paper introduces ExBody, a method enabling humanoid robots to perform diverse motions realistically. Leveraging human motion capture data, ExBody trains a control policy in reinforcement learning. By encouraging the upper body to imitate reference motions while relaxing constraints on leg motion, it achieves robust performance. Through simulation and real-world transfer, ExBody controls robots to walk, handshake, and dance with humans effectively, validated through extensive studies.
   - **Link:** [Expressive Whole-Body Control for Humanoid Robots](https://arxiv.org/pdf/2402.16796.pdf)

## Quantum Computing

### 1. Variational Quantum Circuits for Deep Reinforcement Learning
   - **Authors:** Samuel Yen-Chi Chen, Chao-Han Huck Yang, Jun Qi, Pin-Yu Chen, Xiaoli Ma, Hsi-Sheng Goan
   - **Summary:** This pioneering work explores the integration of variational quantum circuits into deep reinforcement learning algorithms. By reshaping classical deep reinforcement learning techniques such as experience replay and target networks into the framework of variational quantum circuits, the authors aim to address the challenges of leveraging quantum computing in machine learning tasks. The proposed approach demonstrates the first proof-of-principle demonstration of variational quantum circuits for approximating the deep Q-value function, offering promising implications for decision-making and policy selection in reinforcement learning tasks.
   - **Link:** [Variational Quantum Circuits for Deep Reinforcement Learning](https://arxiv.org/abs/1907.00397)

## How to Use

Feel free to explore the papers by clicking on the provided links. Each section is dedicated to a specific field, and within each field, you'll find detailed information about individual papers.
