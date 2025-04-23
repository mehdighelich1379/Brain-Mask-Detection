### MRI Tumor Segmentation using Deep Learning

#### Project Overview:
In this project, the goal was to perform image processing on MRI images of human brains that have tumors. The dataset used is the LGG MRI Segmentation dataset from Kaggle, which contains MRI scans and corresponding masks for segmenting the tumor areas in the images.

---
🔗 Live Demo: Click here to use the app https://brain-mask-detection-with-ai-mehdighelich.streamlit.app

📷 App Preview:


![App Screenshot]()
![App Screenshot]()

### Steps Taken in the Project:

1. Dataset Overview:
   - The dataset consists of MRI images of human brains, specifically of patients with tumors. For each image, there is a corresponding mask that highlights the tumor region. The goal is to segment the tumor from the rest of the brain tissue using deep learning techniques.

2. Loading the Data:
   - I started by reading the images from the folder using cv2 (OpenCV). Each image was resized to 256x256 pixels to ensure uniformity and better performance during training.
   - The mask for each image was also loaded. The mask contains the labeled region of the tumor, where pixels with values greater than 127 were considered part of the tumor and were used for segmentation.

3. Preprocessing the Data:
   - After loading the images and their corresponding masks, I added an extra dimension to each image to make it compatible with the deep learning model. This added an additional channel for the grayscale images, ensuring the model can process the data properly.
   - I also ensured that each image was correctly aligned with its corresponding mask for training the model.

4. Building the Model:
   - For this project, I used a U-Net architecture, a widely used deep learning model for image segmentation tasks. U-Net is especially effective in medical image segmentation due to its encoder-decoder structure, which captures both low-level features and high-level semantic information.
   
5. Training the Model:
   - I trained the model on the preprocessed images and masks, using binary cross-entropy as the loss function since it’s a binary segmentation problem (tumor or non-tumor).
   - The model was trained for several epochs, and I observed that it achieved a 99.59% accuracy on the training data and 99.08% accuracy on the test data.

6. Visualization of Results:
   - I visualized the loss curves (training loss and validation loss) to track the model's convergence and to ensure that the model was not overfitting.
   - I also visualized the accuracy curves (training accuracy and validation accuracy) to monitor how well the model was generalizing on unseen data during the training process.

7. Saving the Model:
   - After training, the best-performing model was saved with the name Mri_segmentation. This model can now be used to segment tumors in new MRI images.

---

### Conclusion:

In this project, I successfully built a deep learning model for tumor segmentation in MRI images using the U-Net architecture. The model achieved 99.59% accuracy on the training data and 99.08% accuracy on the test data, which indicates excellent performance in segmenting tumor regions from the rest of the brain tissue. The model and its training results were visualized through loss and accuracy curves, and the final model was saved for future use.

---

### Skills Demonstrated:
1. Image Preprocessing: Reading and resizing MRI images using cv2, and processing masks for segmentation.
2. Deep Learning: Building a U-Net model for image segmentation, a powerful deep learning architecture for medical images.
3. Model Evaluation: Monitoring accuracy and loss during training to ensure the model was learning effectively and generalizing well.
4. Medical Image Segmentation: Applying deep learning techniques to segment tumor regions from MRI scans.
5. Model Deployment: Saving the trained model for future predictions on new MRI images.

This project demonstrates the application of deep learning to medical image segmentation, specifically for tumor detection in MRI scans, and showcases how powerful these techniques can be in medical diagnostics and research.
