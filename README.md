# Beyond traditional approaches: A multi-task framework for breast cancer segmentation and classification in ultrasound imaging

## Overview
This project presents a novel multi-task framework designed to enhance the accuracy and efficiency of breast cancer diagnosis using ultrasound imaging. Unlike traditional methods that treat segmentation and classification as separate tasks, our approach simultaneously addresses both, leveraging the inherent interdependence between these tasks to improve overall performance. This integrated framework aims to provide more reliable and comprehensive diagnostic tools for clinicians, thereby potentially improving patient outcomes.

![Example Image](./output/images/MT_framework.png)

## Key Features

- **Multi-Task Framework**: Combines breast cancer lesion classification and segmentation tasks to enhance detection performance.
- **Comprehensive Dataset Analysis**: Includes a detailed examination of the BUSI (Breast Ultrasound Images) dataset to identify and address irregularities.
- **Algorithm for Detecting Duplicated Images**: Tailored algorithm to identify and eliminate duplicated images in the dataset, minimizing potential biases.
- **Improved Performance**: Achieves close to 15% improvement in both segmentation and classification tasks compared to single-task approaches.
- **State-of-the-Art Comparison**: Demonstrates statistically significant enhancements over existing methodologies in both tasks.
- **Generalization Capabilities**: Better generalization by considering benign, malignant, and non-tumor images, making it suitable for real clinical applications.


## Experimental Findings

- The multi-task framework significantly outperforms single-task approaches in terms of both segmentation and classification of breast cancer lesions.
- Comprehensive analysis and curation of the BUSI dataset ensure minimized biases and more reliable outcomes.
- The methodology showcases better generalization capabilities, crucial for clinical applications in breast cancer detection.

![Example Image](./output/images/qualitative_results_segmentation.png)

## How to Use

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/multi-task-breast-cancer-detection.git
   cd multi-task-breast-cancer-detection
   

2. **Clone the Repository**

   ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
   
3. **Clone the Repository**: Download the BUSI dataset and place it in the `data/` directory. Run the dataset curation script:

   ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt

4. **Train the Model**: Train the multi-task model using the curated dataset:
   ```bash
   python src/train.py --config configs/multi_task_config.json

5. **Evaluate the Model**: Evaluate the trained model on the test set:
   
   ```bash
   python src/evaluate.py --model models/multi_task_model.pth


## License
This project is licensed under the MIT License. See the LICENSE file for more details.


## Acknowledgments
We thank the contributors of the BUSI dataset and the research community for their invaluable input and support.


---

# Referencing this work
For more information, please refer to our paper.

---

This README file is structured to provide a comprehensive overview of the project, instructions on how to use the repository, and information on how to contribute. If you have any questions or need further assistance, please feel free to contact us.
