R4: RLVR Alignment for Zero-Defect Autoregressive Mesh Generation
The Core Thesis
The evolution of 3D generative AI has been bottlenecked by alignment paradigms that rely on Vision-Language Models (VLMs) to provide aesthetic and semantic supervision. These models yield dense, over-triangulated, or fundamentally flawed meshes characterized by non-manifold edges, pervasive self-intersections, and open boundaries. VLMs are inherently ill-equipped to evaluate the internal mathematical validity of a 3D mesh.
The R4 framework posits a core thesis: an isolated autoregressive generator can be trained to maximize a highly structured, verifiable reward signal to produce zero-defect geometry without any human-in-the-loop or VLM feedback. By subjecting the generative policy to deterministic computational geometry engines (RLVR - Reinforcement Learning with Verifiable Rewards), R4 guarantees that the resulting meshes satisfy strict Boolean topological constraints while maximizing continuous geometric quality metrics.
For a resource-constrained BTech project (Single L4 GPU, 24GB VRAM, 31 hours), proving this core thesis is the primary objective. Advanced, optional features designed to maximize extreme generative diversity (such as Spectral Clustering, Dynamic Sampling, and complex Novelty/Value/Surprise metrics) have been stripped from this implementation plan to ensure feasibility and stable convergence.
Architectural Deliberation: The Autoregressive Base Policy
The designated architecture for this implementation is MeshGPT. MeshGPT formulates 3D mesh generation strictly as a sequence modeling task, utilizing a graph-convolutional autoencoder to learn a discrete vocabulary of latent quantized geometric embeddings.
By utilizing MeshGPT as the frozen reference policy (), the R4 framework inherits a strong geometric prior. Because MeshGPT's pre-training relies solely on maximum likelihood estimation (MLE), it remains susceptible to generating statistically probable but topologically invalid sequences. The RLVR alignment process focuses entirely on refining the probabilities of these discrete tokens to ensure that the decoded sequence unconditionally forms a valid solid.
To fit this within the 24GB VRAM of an L4 GPU, you must use QLoRA and 4-Bit NormalFloat Quantization. The frozen MeshGPT reference policy and the base weights of the active actor policy are quantized to a 4-bit NormalFloat (NF4) data type. High-precision (bfloat16) trainable LoRA adapters are then injected into the linear layers of the transformer's attention and feed-forward modules. This reduces the static weight memory requirement of a 3B parameter model from ~12GB to approximately 2GB.
Simplified RL Optimization: Standard GRPO
Standard Proximal Policy Optimization (PPO) is computationally prohibitive on a single L4 GPU because it requires maintaining a large Critic (Value) network in VRAM.
To solve this, you will implement Group Relative Policy Optimization (GRPO). GRPO fundamentally eliminates the necessity for a distinct value network. Instead, for each prompt or initialization sequence, the model randomly generates a group of  distinct mesh answers.1 The advantage for each individual generation is calculated by normalizing its specific reward against the mean and standard deviation of the entire group's rewards.1
By eliminating the multi-billion parameter value model and calculating relative advantages directly from the sampled group, GRPO slashes the VRAM requirement by up to 50%, making this project viable on your hardware.
Verifiable Reward Engineering
The reward signal must be structurally rigorous, combining absolute binary logic with continuous geometric evaluations to prove the core thesis.
1. Strict Boolean Topological Constraints (The Gatekeepers)
These binary rewards act as absolute computational gatekeepers. You will use deterministic Python libraries to evaluate the decoded meshes:
Watertightness: Use the Trimesh library. A watertight mesh is perfectly closed, edge-manifold, vertex-manifold, and has zero boundary edges. You will use trimesh.is_watertight to evaluate this.2
Self-Intersection: Use the Open3D library. You will use the is_self_intersecting function, which deploys accelerated bounding volume hierarchies to check for face-to-face collisions.3
If a mesh fails either check, it receives a flat reward of 0. If it passes, it receives a high baseline multiplier (e.g., 1.0).
2. Continuous Geometric Quality Metrics
An autoregressive model driven solely by binary checks might learn to generate a featureless, low-polygon sphere just to get a perfect score. To force the model to generate actual, complex shapes, you must add continuous metrics:
Boundary Edge Ratio (BER): Evaluates the proportion of boundary edges relative to the total number of edges.4 This penalizes open surfaces even before they achieve perfect watertightness, providing a continuous learning gradient.
Topology Score (TS): Assesses the structural regularity of the generated triangles to prevent degenerate sliver geometry.4
Safety Framework: Lagrangian Penalty
Balancing continuous aesthetics against hard Boolean topology is a Constrained Markov Decision Process (CMDP). If you simply sum the rewards together, the model might accept a topological failure (reward = 0) if the aesthetic score is exceptionally high.
To fix this without complex engineering, use a simplified Lagrangian Optimization. Treat harmlessness (topological validity) as a hard constraint embedded in the Lagrangian optimization.5 The loss objective is modified to:

Here,  is the Lagrangian multiplier, which acts as a dynamic penalty.5 You can update  using a simple Proportional-Integral (PI) controller after each epoch based on the moving average of the harmlessness cost (i.e., how many meshes are failing the Boolean checks).5 If the model starts generating open meshes,  increases, mathematically forcing the policy back into the safe, zero-defect space.
Concrete Step-by-Step Implementation Procedure (31 Hours)
This roadmap strips away unnecessary algorithmic complexity (like dynamic sampling and spectral clustering) and provides a direct path to proving your thesis.
Step 1: Environment, Dataset, and Baseline Setup (Hours 1 - 5)
Software: Install PyTorch, transformers, and the bitsandbytes library for 4-bit quantization. Install trimesh and open3d for the reward evaluations. For the RL framework, you can utilize veRL, which natively supports efficient GRPO implementations and state-of-the-art throughput.
Dataset Preprocessing: Download a curated dataset of ~1,000 shapes (e.g., a ShapeNet subset of Chairs). Crucial: Preprocess all meshes offline to avoid CPU bottlenecks. Center vertices to , scale them strictly to , and sort vertices by Z, Y, X.
Model Loading: Load the pre-trained MeshGPT model in 4-bit NF4 precision. Inject a LoRA adapter (Rank ) into the attention layers. Confirm that your VRAM usage is under 15GB, leaving room for the generation context window.
Step 2: Reward Engine Construction (Hours 6 - 10)
Write a Python function decode_and_evaluate(token_sequence).
Inside the function, decode the MeshGPT tokens back into a 3D Euclidean mesh.
Implement the Boolean checks:
is_closed = trimesh.is_watertight(mesh)
2
has_collision = open3d.t.geometry.TriangleMesh.is_self_intersecting(mesh)
3
Implement the continuous reward calculations for Boundary Edge Ratio (BER) and Topology Score (TS).4
Set up the reward return: If is_closed is False or has_collision is True, return only the continuous (BER/TS) reward multiplied by a harsh penalty. If True, return the continuous reward + a large Boolean completion bonus.
Step 3: GRPO Loop and Lagrangian Integration (Hours 11 - 15)
Initialize the GRPO trainer. Configure the group size  or  depending on your VRAM limits.
Implement the GRPO advantage calculation: normalize the specific rewards against the mean and standard deviation of the generated group.1
Initialize the Lagrangian multiplier . Write a simple PI controller function that runs at the end of each training step. If the batch failure rate exceeds your tolerance threshold  (e.g., 5%), increase  by a small factor.
Step 4: Active RL Rollout & Fine-Tuning (Hours 16 - 28)
Execute the training loop.
Enable Gradient Checkpointing: You must enable this in PyTorch to discard intermediate activations during the forward pass and recalculate them during the backward pass. This is mandatory to prevent OOM errors on the 24GB L4 GPU.
Monitor the logs closely. You should see the continuous rewards (BER/TS) slowly rise, followed by sharp spikes in the Boolean reward as the model "discovers" how to permanently close the manifolds. Watch the  multiplier; it should rise and fall as the model tests the boundaries of the topological constraints.
Step 5: Benchmarking & Demonstration (Hours 29 - 31)
Freeze the model weights.
Run an inference batch of 100 novel prompts.
Evaluate the output using the exact same deterministic reward functions.
To successfully prove your thesis for your BTech project, generate a final table comparing the baseline pre-trained MeshGPT's topological failure rate (percentage of meshes that are not watertight or have intersections) against your newly aligned R4-MeshGPT model. You should see a dramatic drop in topological defects, proving that RLVR can enforce absolute geometric rules.
Works cited
policy-gradient/GRPO-Zero: Implementing DeepSeek R1's GRPO algorithm from scratch, accessed April 17, 2026, https://github.com/policy-gradient/GRPO-Zero
GitHub - mikedh/trimesh: Python library for loading and using triangular meshes., accessed April 17, 2026, https://github.com/mikedh/trimesh
Mesh — Open3D latest (664eff5) documentation, accessed April 17, 2026, https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html
Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning - OpenReview, accessed April 17, 2026, https://openreview.net/pdf/fbd8d6fd2f8ff032207026054c51e9b9a7bb0100.pdf
Reinforcement Learning for Generative AI: A Survey - arXiv, accessed April 17, 2026, https://arxiv.org/html/2308.14328v3



1. The Base Policy: MeshGPT
Because MeshGPT uses a specialized two-part architecture (a VQ-VAE for geometric tokenization and a Decoder-only Transformer for autoregressive generation), the weights and code are intertwined.

Official Repository (GitHub): audi/MeshGPT

Note: This is the official codebase from the original researchers (Technical University of Munich). You will need to pull their pre-trained VQ-VAE embeddings and baseline transformer weights directly via the download scripts provided in their repository, as they do not host a standard one-click model card on Hugging Face.

Community PyTorch Port (GitHub): Yuan-ManX/MeshGPT-PyTorch

Note: If you run into environment issues with the original code, this is a popular, clean PyTorch reimplementation that might be easier to integrate into your veRL pipeline.

2. The Dataset: ShapeNetCore (Chair Subset)
ShapeNet is the gold standard for 3D generation tasks, but it is heavily regulated. You cannot simply wget it without authorization.

Official Hugging Face Dataset: ShapeNet/ShapeNetCore

GLB Format Version: ShapeNet/shapenetcore-glb (This version might be easier to parse directly into trimesh and open3d than the raw .obj files).

How to Access: ShapeNet is a gated dataset. You must create a Hugging Face account, navigate to one of the links above, fill out the academic request form (use your university email), and agree to their non-commercial terms. Approval is usually instantaneous.

The Subset (Chairs): Once downloaded, the dataset is organized by WordNet synset IDs. The exact folder/synset ID for the Chairs subset is 03001627. You will find roughly ~6,700 chair models in this directory. You can easily sample your required ~1,000 shapes from here.