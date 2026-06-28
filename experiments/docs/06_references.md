# References

## Primary Papers

1. **PoseMamba** — Huang et al., "PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model", AAAI 2025. [arXiv:2408.03540](https://arxiv.org/abs/2408.03540)
   - Our baseline. Bidirectional global-local SSM for 3D HPE.
   - Code: https://github.com/xiWang - later commit.

2. **PoseMagic** — "PoseMagic: Dual-stream GCN-Mamba for 3D Human Pose Estimation", AAAI 2025.
   - GCN+Mamba dual-stream with adaptive fusion. Δ = -0.9mm over PoseMamba.
   - **H1 evidence (Level 5).**

3. **SasMamba** — "SasMamba: Structure-Aware Stride State Space Model for 3D Human Pose Estimation", WACV 2026.
   - Structure-aware stride scan over skeleton partitions.
   - **H2 evidence (Level 3).**

4. **DBMambaPose** — "DBMambaPose: Decoupled Bidirectional Mamba for 3D Human Pose Estimation", arXiv 2025.
   - Decoupled spatial-temporal scans with independent SSM parameters.
   - **H5 evidence (Level 3).**

5. **Mamba-Driven Topology Fusion** — "Mamba-Driven Topology Fusion for 3D Human Pose Estimation", arXiv 2026.
   - Bone-aware module with direction/length vectors.
   - **H6 evidence (Level 3).**

## SSM Theory

6. **Mamba-1** — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
   - Selective SSM foundation. State dim N=16.

7. **Mamba-2** — Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
   - SSD layer: scalar A, larger state dim (N=64+), chunked training.
   - **H9 evidence (Level 4).**

8. **Mamba-3** — Lahoti et al., "Mamba-3: Overcoming the Mamba-Kernel Paradox", 2026.
   - Complex-valued states, MIMO SSM, RoPE, QKNorm.
   - **H10 evidence (Level 3).**

9. **S4** — Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022.
   - Original structured state space model.

10. **S6** — Selective scan algorithm used in Mamba-1.

## Hybrid SSM-Attention

11. **Jamba** — Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", AI21 Labs, 2024. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
    - 93% SSM + 7% attention layers.

12. **TransMamba** — "TransMamba: Fast Universal Architecture via Hybrid Transformer and Mamba", arXiv 2025.
    - Dual-path SSM + attention.

13. **Nemotron** — NVIDIA, "Nemotron-4 340B Technical Report", 2025.
    - Production hybrid (95% SSM + 5% attention).

## Vision SSMs

14. **VMamba** — Liu et al., "VMamba: Visual State Space Model", 2024. [arXiv:2401.10166](https://arxiv.org/abs/2401.10166)
    - 2D cross-scan for image processing.

15. **Spatial-Mamba** — "Spatial-Mamba: Spatial State Space Model for Visual Understanding", ICLR 2025.
    - 3×3 depthwise conv replaces 1D causal conv.

16. **ASGMamba** — "ASGMamba: Adaptive Spectral Gating Mamba for Visual Recognition", arXiv 2026.
    - Frequency-selective gating via patch-level FFT.

17. **LocalViM** — "LocalViM: Local Vision Mamba with Windowed Scanning", arXiv 2025.
    - Windowed local scanning for vision.

## Training & Methodology

18. **Cosine Annealing** — Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017.

19. **AdamW** — Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019.

20. **Sapiens** — Khirodkar et al., "Sapiens: Foundation for Human Vision Models", CVPR 2024.
    - Training recipe: cosine warmup, gradient clipping, EMA.

21. **MotionBERT** — Zhu et al., "MotionBERT: Unified Pretraining for Human Motion Analysis", ICCV 2023.

22. **MotionAGFormer** — Ma et al., "MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network", WACV 2024.

## Other SSM Architectures

23. **MambaMixer** — "MambaMixer: Efficient Selective State Space Models with Mixture of Experts", 2024.

24. **Gated DeltaNet** — Yang et al., "Gated DeltaNet: A Gated Linear Recurrent Unit for Language Modeling", 2024.

25. **MambaByte** — "MambaByte: Token-free SSM Language Model", 2024.

## Hybrid Mamba Architectures

26. **VIMCAN** — Yang et al., "VIMCAN: Visual-Inertial 3D Human Pose Estimation with Hybrid Mamba-Cross-Attention Network", CVPR 2026. [arXiv:2605.07552](https://arxiv.org/abs/2605.07552)
    - First Mamba+Cross-Attention hybrid for HPE. Proves hybrid works for pose.

27. **SAMA** — Lu et al., "A Structure-aware and Motion-adaptive Framework for 3D Human Pose Estimation with Mamba", ICCV 2025.
    - Structure-aware State Integrator + Motion-adaptive State Modulator. State-level topology fusion.
    - **H11 evidence (Level 3).**

28. **AGMamba** — "AGMamba: Monocular 3D human pose estimation with a spatio-temporal Mamba-AGCN network", Signal Processing: Image Communication, 2026.
    - Mamba + Attention-GCN dual-stream with adaptive fusion.

29. **HGMamba** — "HGMamba: Enhancing 3D Human Pose Estimation with a HyperGCN-Mamba Network", arXiv 2025. [arXiv:2504.06638](https://arxiv.org/abs/2504.06638)
    - HyperGCN + Mamba dual-stream. 38.65mm P1 on H36M.

## Production-Grade Hybrid SSM-Attention (NLP)

30. **Jamba** — Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", AI21 Labs, ICLR 2025. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
    - 93% SSM + 7% attention. Production-grade. **Key finding: Mamba-1+Attention > Mamba-2+Attention.**

31. **Jamba-1.5** — Jamba Team, "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale", AI21 Labs, 2025. [arXiv:2408.12570](https://arxiv.org/abs/2408.12570)
    - 94B active params, 398B total. 256K context. MoE + hybrid.

32. **TransMamba** — Li et al., "TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model", AAAI 2026.
    - Shared QKV and CBx parameters. Dynamic switching between attention and SSM.

33. **Nemotron-H** — NVIDIA, "Nemotron-H: Hybrid Mamba-Transformer Models", 2025.
    - 92% Mamba2 + 8% attention. 3× faster than LLaMA-3.1. Open source.

34. **Bamba-9B** — IBM, "Bamba-9B: Open-Source Hybrid Mamba-Transformer Model", 2025.
    - Mamba2 + Transformer. 2× throughput. Matches LLaMA-3.1-8B with 7× less data.

## Datasets

35. **Human3.6M** — Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments", TPAMI 2014.

36. **MPI-INF-3DHP** — Mehta et al., "Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision", 3DV 2017.

37. **3DPW** — von Marcard et al., "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera", ECCV 2018.

38. **EMDB** — "EMDB: The Electromagnetic Database of 3D Human Pose and Shape", 2022.
