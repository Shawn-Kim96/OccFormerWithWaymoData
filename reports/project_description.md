This assignment was locked Sep 21 at 11:59pm.
The goal of the project proposal is to clearly define the topic, motivation, goals, and technical plan for your semester-long project. You will choose Option 1 (Apply & Evaluate) or Option 2 (Fine-Tune & Train) as your project track. This proposal helps ensure your project is feasible, well-scoped, and technically meaningful.

Option 1 (Apply & Evaluate): Apply and Evaluate Pretrained Models (No Training)
Deploy, apply, and analyze existing pretrained deep learning models for autonomous driving tasks (e.g., perception, BEV detection, occupancy network, 2D/3D fusion, path prediction) in ROS 2-based environments ( or Nvidia Isaac ROS), including simulation (Nvidia Isaac Sim) or real-world datasets (NuScenes, Waymo, Argo).
Requirements
Select a state-of-the-art pretrained model, understand the model’s structure, inputs, outputs (Evaluation: presentation)
Wrap the model into a ROS 2 node with clearly defined input/output topics.
Apply the model to a ROS 2-based simulation or real-world dataset.
Evaluate model accuracy, measure system-level latency and resource utilization (CPU/GPU/Memory) under different computing environments (e.g., Jetson, laptop, server).
Option 2 (Fine-Tune & Train): Fine-Tune and Extend Models (Training Required)
Fine-tune or adapt an existing autonomous driving model on a new dataset (NuScenes, Waymo, Argo or simulated data from Nvidia Isaac Sim) and analyze the improvements and tradeoffs in performance and efficiency.
Requirements
Select a state-of-the-art pretrained model, understand the model’s structure, inputs, outputs (Evaluation: presentation)
Fine-tune the model with your selected dataset.
Experiment with multiple configurations (e.g., learning rates, data augmentations, model variants) or modify model architecture.
Compare: pretrained vs. fine-tuned performance; performance across multiple training configurations.
Evaluate runtime, model size, latency, and resource usage on different platforms (e.g., server vs. embedded).
Submission Format: PDF file via Canvas
Proposal Content (2–4 pages)
Project Title and Team Info
Project Option Selection: Option1 or Option2
Motivation and Use Case: What is the driving problem or use case
Selected Model(s): Name and source of the model(s) you plan to use; brief explanation of what the model does; link to official GitHub or paper
Dataset and Input/Output: What dataset(s) will you use? (real-world or simulated)
Timeline (Milestones): Provide a week-by-week plan with key milestones.