# Visual Agents: What it Takes to Build an Agent that can Navigate GUIs like Humans

## ODSC Workshop

This workshop provides a hands-on introduction to building visual agents for GUI automation using [FiftyOne](https://voxel51.com/fiftyone/) for dataset curation, visualization, and model evaluation.

### What You'll Learn

- **GUI Dataset Creation**: How to structure and annotate GUI interaction data
- **Data Exploration**: Using FiftyOne to visualize and analyze GUI datasets
- **Embeddings & Similarity**: Computing multimodal embeddings for UI elements
- **Model Inference**: Running GUI-Actor predictions on your data
- **Evaluation & Debugging**: Identifying model failures and tagging samples for improvement
- **Data Augmentation**: Using FiftyOne plugins to generate synthetic training samples
- **PyTorch Integration**: Converting FiftyOne datasets for model training

### Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `gui_agents_workshop.ipynb` and follow along

### Workshop Materials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harpreetsahota204/odsc_workshop_visual_agents/blob/main/gui_agents_workshop.ipynb)

[![Google Slides](https://img.shields.io/badge/Google%20Slides-View%20Presentation-blue?logo=google&logoColor=white)](https://docs.google.com/presentation/d/1I-g2R9Jm_eblzfRrDJtXgbg-gr_KyoZb4IG1bYUa1W8/edit?usp=sharing)

---

## Resources

This workshop is an abridged version of a longer, more comprehensive workshop series on building visual agents. Below are resources from the full workshop for those who want to dive deeper.

### Full Workshop Series

[![GitHub Stars](https://img.shields.io/github/stars/harpreetsahota204/visual_agents_workshop?style=social)](https://github.com/harpreetsahota204/visual_agents_workshop)

The complete Visual Agents Workshop Series provides a progressive learning path through GUI agent research, data preparation, and model implementation.

---

### Session 1: Navigating the GUI Agent Research Landscape

An exploration of GUI agent research from 2016-2025, providing a data-driven foundation for understanding the field's evolution.

**Key Topics:**
- Research methodology for analyzing academic papers
- Trend analysis of GUI agent development
- Citation networks and influential research
- Field trajectory and future directions

**Resources:**

- [![YouTube](https://img.shields.io/badge/YouTube-Watch%20Session-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=d-bLgV3GFqE&t=1s) [![Watch on YouTube](https://img.shields.io/badge/Watch-on%20YouTube-red?style=flat&logo=youtube)](https://www.youtube.com/watch?v=d-bLgV3GFqE&t=1s)

- [![Google Slides](https://img.shields.io/badge/Google%20Slides-View%20Presentation-blue?logo=google&logoColor=white)](https://docs.google.com/presentation/d/1yoZmles5Do4y_-mubmnWYItx7Pe_3CwHuCQG6Uu11zQ/edit?usp=sharing)

- [![GitHub](https://img.shields.io/badge/GitHub-Research%20Repository-black?logo=github&logoColor=white)](https://github.com/harpreetsahota204/gui_agent_research_landscape) [![Stars](https://img.shields.io/github/stars/harpreetsahota204/gui_agent_research_landscape?style=social)](https://github.com/harpreetsahota204/gui_agent_research_landscape)

---

### Session 2: From Pixels to Predictions - Building GUI Datasets

A practical deep dive into creating and managing datasets for training visual agents.

**Key Topics:**
- Understanding diverse GUI agent datasets
- Dataset challenges and training pipeline complexity
- COCO4GUI standardized format
- Generating synthetic data for robustness
- Data augmentation techniques

**Resources:**

- [![YouTube](https://img.shields.io/badge/YouTube-Watch%20Session-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=mCBJHQ5SYJg) [![Watch on YouTube](https://img.shields.io/badge/Watch-on%20YouTube-red?style=flat&logo=youtube)](https://www.youtube.com/watch?v=mCBJHQ5SYJg)

- [![Google Slides](https://img.shields.io/badge/Google%20Slides-View%20Presentation-blue?logo=google&logoColor=white)](https://docs.google.com/presentation/d/1KIrjqpvQ9bNa0Dv_wN3UetStCijn9Uf5rf1xjh6gJZA/edit?usp=sharing)

- [![Jupyter](https://img.shields.io/badge/Jupyter-Workshop%20Notebook-orange?logo=jupyter&logoColor=white)](https://github.com/harpreetsahota204/visual_agents_workshop/blob/main/session_2/working_with_gui_datasets.ipynb)

- [![GitHub](https://img.shields.io/badge/GitHub-COCO4GUI%20Dataset%20Creator-black?logo=github&logoColor=white)](https://github.com/harpreetsahota204/gui_dataset_creator)

- [![GitHub](https://img.shields.io/badge/GitHub-FiftyOne%20Dataset%20Importer-black?logo=github&logoColor=white)](https://github.com/harpreetsahota204/coco4gui_fiftyone)

- [![GitHub](https://img.shields.io/badge/GitHub-Synthetic%20GUI%20Samples%20Plugin-black?logo=github&logoColor=white)](https://github.com/harpreetsahota204/synthetic_gui_samples_plugins)

---

### Session 3: Teaching Machines to See and Click - Model Finetuning

An in-depth exploration of model architectures and fine-tuning strategies for GUI agents.

**Key Topics:**
- GUI agent model architectures (OS Atlas, MiMo-VL, ShowUI-2B, GUI-Actor)
- The role of Qwen VL as a foundational backbone
- Fine-tuning strategies and methodologies
- Coordinate system standardization
- Model evaluation and benchmarking

**Resources:**

- [![YouTube](https://img.shields.io/badge/YouTube-Watch%20Session-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Lzl-ktGbAz8) [![Watch on YouTube](https://img.shields.io/badge/Watch-on%20YouTube-red?style=flat&logo=youtube)](https://www.youtube.com/watch?v=Lzl-ktGbAz8)

- [![Google Slides](https://img.shields.io/badge/Google%20Slides-View%20Presentation-blue?logo=google&logoColor=white)](https://docs.google.com/presentation/d/1qF6BEevIyYLWENLAxh8dCJFw4uRfe54MvCc-xh8Rkwc/edit?usp=sharing)

- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Training%20Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/harpreetsahota/FiftyOne-GUI-Grounding-Train-with-Synthetic)

- [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Benchmark%20Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/harpreetsahota/FiftyOne-GUI-Grounding-Eval)

- [![GitHub](https://img.shields.io/badge/GitHub-GUI--Actor%20Finetuning%20Repository-black?logo=github&logoColor=white)](https://github.com/harpreetsahota204/GUI-Actor-for-FiftyOne)

- [![Jupyter](https://img.shields.io/badge/Jupyter-Testing%20Notebook-orange?logo=jupyter&logoColor=white)](https://github.com/harpreetsahota204/visual_agents_workshop/blob/main/session_3/Testing_GUI_Actor_3B_on_FiftyOne_App_Dataset.ipynb)

- [![Jupyter](https://img.shields.io/badge/Jupyter-Finetuning%20Notebook-orange?logo=jupyter&logoColor=white)](https://github.com/harpreetsahota204/visual_agents_workshop/blob/main/session_3/Fine_tuning_GUI_Actor_3B_on_FiftyOne_App_Dataset_.ipynb)

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
