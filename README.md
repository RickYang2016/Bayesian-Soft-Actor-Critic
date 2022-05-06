# Bayesian-Soft-Actor-Critic

  Adopting reasonable strategies is challenging but crucial for an intelligent agent with limited resources working in hazardous, unstructured, and dynamic changing environments to improve the system utility, decrease the overall cost, and increase mission success probability. Deep Reinforcement Learning (DRL) helps organize agents' behaviors and actions based on their state and represents complex strategies (composition of actions). This project proposes a novel hierarchical strategy decomposition approach based on Bayesian chaining to separate an intricate policy into several simple sub-policies and organize their relationships as Bayesian strategy networks (BSN). We integrate this approach into the state-of-the-art DRL method, soft actor-critic (SAC), and build the corresponding Bayesian soft actor-critic (BSAC) model by organizing each sub-policies as a joint policy.

![image](https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d.png)

![image](https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/policy_network.png)

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2_3bsac.png" height="250" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2_3bsac.gif" height="230" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/hopper-v2.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>
  
<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d_v2_5bsac.png" height="215" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d-v2_5bsac.gif" height="230" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/walker2d-v2.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2_5bsac.png" height="177" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2_3bsac.gif" height="230" alt="Hopper-V2 3SABC Video"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2-compare.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>

<div align = center>
<img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/humanoid-v2-3%269bsac.png" height="230" alt="Hopper-V2 3SABC"><img src="https://github.com/RickYang2016/Bayesian-Soft-Actor-Critic/blob/main/figures/bsac_compare.png" height="250" alt="Hopper-V2 3SABC Video"/>
</div>
