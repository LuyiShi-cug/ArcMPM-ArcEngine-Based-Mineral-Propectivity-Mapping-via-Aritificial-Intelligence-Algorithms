
<style>
.center { text-align: center; }
</style>
<h1 class="center">ArcMPM V2.0</h1>

[TOC]

# 1 Software Introduction

The Intelligent Mineral Prospectivity Mapping Software (ArcMPM, ArcEngine-based software for mineral prospectivity mapping (MPM) via artificial intelligence (AI) algorithms) is developed using C# and Python languages on the ArcEngine platform. ArcMPM mainly includes functions such as geological data visualization, negative points selection, train sample generation, AI-based geochemical anomaly recognition, and AI-based mineral prospectivity mapping, aiming to provide a complete workflow for intelligent mineral exploration.

# 2 Installation and Uninstallation

## 2.1 Software Operating Requirements

This software is an application developed based on the ArcEngine platform. The operating system environment must be Windows 7 or later versions. Additionally, a license of ArcGIS 10.2 for Desktop or higher is required.

## 2.2 Software Installation

(1) First, open the software installer ArcMPM.msi (Figure 1), and follow the installation wizard by clicking Next.  
![placeholder](/figure/1.png)  
Figure 1 Installer  

(2) After selecting the installation folder, click Next (Figure 2).  
(3) Then click Next again to continue the installation (Figure 3).  
(4) Wait for the software installation to complete, then close the installer.  
(5) Extract the previously downloaded Python 3.9 compressed package into the ArcMPM installation directory (Figure 4).  
(6) The software installation is complete. Run the software ArcMPM (Figure 5).

![placeholder](/figure/2.png)
图 2选择安装路径

![placeholder](/figure/3.png)
图 3安装进度

![placeholder](/figure/4.png)
图 4放置Python39文件夹

![placeholder](/figure/5.png)
图 5软件界面

## 2.3 Software Uninstallation

Click Uninstall.exe in the directory, select "Yes" to start the removal process, and then manually delete the Python3.9 folder in the directory to completely uninstall the software (Figure 6).

![placeholder](/figure/6.png)
图 6卸载软件

# 3 Data Loading

Click [File] → [Open Mxd File] to select and open a map document; click [File] → [Open Shp File], [File] → [Open Tif File], or [File] → [Open Grid File] to add data through these options. You can also open map documents and add data by clicking ![placeholder](/figure/openfileico.png).  
Here, we take opening a map document as an example. Click [File] → [Open Mxd File]. In the pop-up dialog box, select the map document and then click Open (Figure 7).

![placeholder](/figure/7.png)
Figure 7 Data Loading

# 4 Sample Making

## 4.1 Negative Points Selection

Click [Sample Making] → [Negative Points Selection] to open the negative points selection interface (Figure 8).  
![placeholder](/figure/8.png)  
Figure 8 Negative Points Selection Interface

- **Positive Points Path:** Use known mineral sites as positive points. You can add shapefiles already loaded into the software via the dropdown menu, or click the ![placeholder](/figure/openfileico.png) button to select files.
- **Processing Area:** This is the study area mask and coordinate information for the entire research region. Raster data must be used as input here. In this example, any evidence layer of the study area can be used. You can select loaded raster data from the dropdown menu or click the ![placeholder](/figure/openfileico.png) button to choose a file.
- **Constraint Entity:** Defines the area restricting where negative points can appear, limiting the locations of generated negative points. You can select loaded data from the dropdown menu or click the ![placeholder](/figure/openfileico.png) button to choose a file.
- **Constraint Entities:** A collection of constraint areas limiting where negative points can appear. After specifying a constraint area, click the [↓] button to add it to the set. By default, negative points will appear within 1000 meters of the constraint area. To remove an item from the set, select the entire row and click the [×] button. To modify the distance from the constraint area, click the distance value in that row and enter the desired distance in meters.
- **Negative Points Layer Output Path:** Choose the output path for the generated negative points file, which will be a shapefile. Click the ![placeholder](/figure/openfileico.png) button to select the directory.

After entering all parameters, click [Generate] and wait for the process to complete. The generated negative  points will be directly loaded into the software (Figure 9).  

![placeholder](/figure/9.png)  
Figure 9 Negative Points Display

## 4.2 Training Sample Preparation

If subsequent geochemical anomaly identification is needed, the ore-control layer input should be geochemical data. Conversely, if mineral prospectivity mapping is required, the ore-control layer input can include various sources of mineral exploration data, but the data format must be raster layers with the same extent and cell size.

### 4.2.1 Supervised Dataset Construction

Select [Sample Making] → [Training Sample Construction] → [Supervised Sample Construction] to open the supervised dataset construction interface (Figure 10).  
![placeholder](/figure/10.png)  
Figure 10 Supervised Dataset Construction Interface

- **Positive Points Layer:** Use known mineral deposit sites as positive points. You can add shapefiles already loaded into the software via the dropdown menu or click the button to select files.
- **Negative Points Layer:** The negative points generated in [Sample Making] → [Negative Points Selection]. You can select negative points layer directly from the dropdown menu or click the button to select files.
- **Ore-control Layer:** Layers related to controlling mineralization factors, in raster format. You can add raster files already loaded into the software via the dropdown menu or click the button to select files.
- **Ore-control Layers:** A collection of layers related to controlling mineralization factors. This interface will by default load raster data already imported into the software as ore-control layers into this list. Users can also add or remove data manually.
- **Augment Size:** The window size used for data augmentation; Input must be an odd integer.
- **Window Size:** If you need to build samples for convolutional neural network (CNN) training, check this option and enter the desired window size; Input must be an odd integer.
- **Distance for Edge:** If you need to build samples for graph neural network (GNN) training, check this option and enter the Euclidean distance used to construct the graph; Input must be an integer.
- **Dataset Save Path:** The directory where the supervised learning dataset will be generated (using an empty folder is recommended). Click the ![placeholder](/figure/openfileico.png) button to select the directory.

After entering all parameters, click [Generate] to create the samples in the specified directory (Figure 11). Upon successful generation, a confirmation message will appear.  
![placeholder](/figure/11.png)  
Figure 11 Supervised Learning Dataset

### 4.2.2 Unsupervised Dataset Construction

Select [Sample Making] → [Training Sample Construction] → [Unsupervised Sample Construction] to open the unsupervised dataset construction interface (Figure 12).  

![placeholder](/figure/12.png)  
Figure 12 Unsupervised Dataset Construction Interface

- **Ore-control Layer:** Layers related to controlling mineralization factors, in raster format. You can add raster files already loaded into the software via the dropdown menu or click the button to select files.
- **Ore-control Layers:** A collection of layers related to controlling mineralization factors. This interface will by default load raster data already imported into the software as ore-control layers into this list. Users can also add or remove data manually.
- **Augment Size:** The window size used for data augmentation; Input must be an odd integer.
- **Window Size:** If you need to build samples for convolutional neural network (CNN) training, check this option and enter the desired window size; Input must be an odd integer.
- **Distance for Edge:** If you need to build samples for graph neural network (GNN) training, check this option and enter the Euclidean distance used to construct the graph; Input must be an integer.
- **Dataset Save Path:** The directory where the unsupervised learning dataset will be generated (using an empty folder is recommended). Click the ![placeholder](/figure/openfileico.png) button to select the directory.

After entering all parameters, click [Generate] to create the samples in the specified directory. Upon successful generation, a confirmation message will appear (Figure 13).

![placeholder](/figure/13.png)  
Figure 13 Unsupervised Learning Dataset

# 5 Geochemical Anomaly Identification

## 5.1 Spectral-Spatial Dual-Branch Model

Select [Geochemical Anomaly Identification] → [Spatial-Spectral Dual-Branch Model] to open the dual-branch model interface (Figure 14).

![placeholder](/figure/14.png)  
Figure 14 Dual-Branch Model Interface – Model Training

- **Load Dataset:** Directory containing dataset generated for supervised learning. Here, you need to select the file path where the data is stored. You can use ![placeholder](/figure/openfileico.png) to choose the file path.
- **Loss Function:** Used to measure the inconsistency between the model’s predicted values and the true values. The loss function guides the model’s learning. The software provides the cross-entropy loss function; no user adjustment is needed.
- **Epochs:** The number of training iterations for the model. Input should be an integer. Users can adjust this as needed.
- **Learning Rate:** When optimizing using gradient descent, the gradient term in the weight update rule is multiplied by a coefficient called the learning rate. Input should be a non-zero decimal. Users can adjust this as needed.
- **Optimizer:** The optimizer used during model training. The software only provides Adam; no user adjustment is needed.
- **Model Save Path:** The path where the trained model will be saved. You can use ![placeholder](/figure/openfileico.png) to select the save path.
- **Model Name:** The filename for the trained model. Users can input this freely.

After setting the parameters, you can click [Train] to train and predict using the default network structure, or click [Model Design] to customize the network architecture.

Here, we use a custom network architecture for model training as an example by clicking [Model Design] (Figure 15). When setting up the spectral-spatial dual-branch network, priority should be given to configuring the spatial branch and spectral branch model structures first, followed by the fusion model structure. The “Window Size” parameter corresponds to the “window size” set during sample preparation and should be adjusted according to the actual sample window size.

![placeholder](/figure/15.png)  
Figure 15 Dual-Branch Model Interface – Spatial Model Design

The following network layers are available for the spatial branch model:

- **2D Convolutional Neural Network (2DCNN):**  
  - **Input Size:** The input size of the first layer should match the number of geochemical element types. This is automatically calculated by the software and does not require manual adjustment.  
  - **Output Size:** Users can set the output size of this layer as needed. Input should be an integer.  
  - **Activation Function:** The software provides ReLU and GELU activation functions. Users can select via dropdown.  
  - **Padding:** Number of padding layers around the data before convolution. Users can set as needed.  
  - **Kernel Size:** Size of the convolution kernel during convolution operations. Users can set as needed.  
  - **Stride:** The movement step size of the convolution kernel during convolution. Users can set as needed.  
  - **Dropout:** Adds a Dropout layer after this layer to deactivate a proportion of neurons. Input should be a decimal between 0 and 1. Users can adjust as needed.

- **Pooling:**  
  - **Pooling Type:** The software provides Average Pooling and Max Pooling. Users can select as needed.  
  - **Pooling Size:** The window size of the pooling layer. Input should be an integer.  
  - **Pooling Stride:** The stride of the pooling window movement. Input should be an integer.

- **Graph Convolutional Layer (GCN):**  
  - **Input Size:** Matches the number of geochemical element types, auto-calculated by software.  
  - **Output Size:** User-defined integer.  
  - **Activation Function:** ReLU or GELU selectable via dropdown.  
  - **Dropout:** Decimal between 0 and 1, adjustable.

- **Graph Attention Layer (GAT):**  
  - **Input Size:** Matches the number of geochemical element types, auto-calculated.  
  - **Output Size:** User-defined integer.  
  - **Activation Function:** ReLU or GELU selectable.  
  - **Dropout:** Decimal between 0 and 1.  
  - **Head:** Number of Attention Heads.Integer, adjustable.

![placeholder](/figure/16.png)  
Figure 16 Dual-Branch Model Interface – Spatial Model Design – Spectral Model Design

The spectral branch offers the following network layers (Figure 16):

- **1D Convolutional Neural Network (1DCNN):**  
  - **Input Size:** Matches the number of geochemical element types, auto-calculated.  
  - **Output Size:** User-defined integer.  
  - **Activation Function:** ReLU or GELU selectable.  
  - **Dropout:** Decimal between 0 and 1.

- **Recurrent Neural Network (RNN):**  
  - **Input Size:** Matches the number of geochemical element types, auto-calculated.  
  - **Output Size:** User-defined integer.  
  - **Activation Function:** ReLU or GELU selectable.  
  - **Dropout:** Decimal between 0 and 1.
  - **RNN Type:** Choose the RNN layer type to add—options include vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). Select via dropdown.

- **Linear Layer (Linear):**  
  - **Input Size:** Matches the number of geochemical element types, auto-calculated.  
  - **Output Size:** User-defined integer.  
  - **Activation Function:** ReLU or GELU selectable.  
  - **Dropout:** Decimal between 0 and 1.

![placeholder](/figure/17.png)  
Figure 17 Dual-Branch Model Interface – Spatial Model Design – Fusion Model Design

Only Linear are available for the fusion model (Figure 17). It is recommended to set the dropout rate to 0 and the output dimension of the last layer to 2, with the activation function set to softmax.

After completing all network settings, click [Save] to use the configured network structure for training. Click [Train] to start training; a progress bar will appear. When training is complete, loss and accuracy curves during training will be displayed (Figure 18).

![placeholder](/figure/18.png)  
Figure 18 Dual-Branch Model Training Process Curve

After closing the training process curves, evaluation metric charts for the model will appear (Figure 19), including Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

![placeholder](/figure/19.png)  
Figure 19 Dual-Branch Model Evaluation Metrics

After closing the evaluation metrics, the model will be used for prediction, and the prediction results will be directly loaded into the software (Figure 20). All generated charts and the model from the training process will be saved in the “Model Save Path.”

![placeholder](/figure/20.png)  
Figure 20 Dual-Branch Model Results

## 5.2 Spectral-Spatial Dual-Branch Autoencoder

Select [Geochemical Anomaly Identification] → [Spatial-Spectral Dual-Branch Autoencoder Model] to open the dual-branch autoencoder construction interface (Figure 21).

![placeholder](/figure/21.png)  
Figure 21 Dual-Branch Autoencoder Interface

- **Dataset Path:** Directory containing dataset generated for unsupervised learning samples. Select the file path where data is stored using ![placeholder](/figure/openfileico.png).
- **Loss Function:** Measures the inconsistency between predicted and true values. The software only provides the Mean Square Error (MSE) loss function; no user adjustment is needed.
- **Epochs:** Number of training iterations. Input should be an integer.
- **Learning Rate:** Coefficient multiplied by the gradient term during weight updates in gradient descent optimization. Input should be a non-zero decimal.
- **Optimizer:** Optimizer used for training the dual-branch model. Only Adam is provided.
- **Model Save Path:** Path to save the trained model. Selectable via ![placeholder](/figure/openfileico.png).
- **Model Name:** Filename for the trained model. User-defined.

After setting parameters, click [Train] to train and predict using the default network structure, or click [Model design] to customize the network architecture. In this interface, users can customize the encoder network structures for the spatial and spectral branches. The model will automatically generate corresponding decoder structures based on the user-defined encoder. The available network layer types and operations are consistent with those in the dual-branch network described above and will not be repeated here.

After completing all network settings, click [Save] to use the configured network structure for training. Click [Train] to start training; a progress bar will appear. When training is complete, the model will be used for prediction, and the prediction results will be directly loaded into the software (Figure 22). The trained model during training will be saved in the “Model Save Path.”

![placeholder](/figure/22.png)  
Figure 22 Dual-Branch Autoencoder Results

# 6 Mineral Prospectivity Mapping

## 6.1 Supervised Learning Algorithm

### 6.1.1 Random Forests

Click [Mineral Prospectivity Mapping]→[Supervised Learning Algorithm]→[Random Forests] to open the Random Forests interface(Figure 23).

![placeholder](/figure/23.png)
Figure 23 Random Forests Interface

- **Dataset Path:** Select the directory containing the dataset generated for supervised learning samples. Here, you need to choose the file path where the data is stored. You can select the file path by clicking ![placeholder](/figure/openfileico.png).
- **Proportion of Training and Validation Set:** Choose the ratio for splitting the training and test sets via the dropdown menu.
- **N_Estimators:** The number of decision trees to build in the random forests. Input type is integer; users can adjust as needed.
- **Maximum Depth:** The maximum depth of each tree. Input type is integer; users can adjust as needed.
- **Maximum Features:** The maximum number of features used by each tree. Select via dropdown or manually input a decimal between 0 and 1.
- **Criterion:** The software provides two options, gini and entropy, selectable via dropdown.
- **Min Samples Leaf:** Input type is integer; users can adjust as needed.
- **Min Samples Split:** Input type is integer; users can adjust as needed.
- **Model Save Path:** The output path for the trained model. Select the file path by clicking ![placeholder](/figure/openfileico.png).
- **Model Name:** The filename for the trained model. You can use the default name or input your own.
After setting all parameters, click the [Train] button to start training the random forests model. Upon completion, evaluation metrics will pop up (Figure 24). After closing this window, the result map will be generated and loaded directly into the software (Figure 25).
![placeholder](/figure/24.png)
Figure 24 Random Forest Evaluation Metrics
![placeholder](/figure/25.png)
Figure 25 Random Forest Prediction Result Map

### 6.1.2 Convolutional Neural Network

Click [Mineral Prospectivity Mapping]→[Supervised Learning Algorithm]→[Convolutional Neural Network] to open the CNN model construction window (Figure 26).

![placeholder](/figure/26.png)
Figure 26 CNN Model Construction Interface - Model Training

- **Dataset Path:** Select the directory containing dataset generated for supervised learning samples (make sure to check "Window Size"). Choose the file path by clicking ![placeholder](/figure/openfileico.png).
- **Loss Function:** Measures the discrepancy between predicted and true values. The software provides cross-entropy loss, which users do not need to adjust.
- **Epochs:** The number of training iterations. Input type is integer; users can adjust.
- **Learning Rate:** The coefficient multiplied by the gradient during weight updates in gradient descent optimization. Input type is a non-zero decimal; users can adjust.
- **Optimizer:** The optimizer used during training. The software provides Adam; no adjustment needed.
- **Model Save Path:** Path to save the trained model. Select via ![placeholder](/figure/openfileico.png).
- **Model Name:** Filename for the trained model; users can input.

After setting parameters, you can click [Train] to train and predict using the default network structure, or click [Model Design] to customize the network architecture.
As an example, click [Model Design] to customize the network layers (Figure 27). When designing the network, first set convolutional and pooling layers for spatial feature extraction, then add fully connected layers for classification. The last layer must be a fully connected layer with output dimension 2 and softmax activation. The “Window Size” parameter corresponds to the sample window size set during sample preparation and should be adjusted accordingly. Model parameters refer to explanations in section 5.1 regarding convolutional and fully connected layers.

![placeholder](/figure/27.png)
Figure 27 CNN Construction Interface - Model Structure Design

After design, click [Save] to save the model structure, then click [Train] to train and predict using the custom network. Upon training completion, loss and accuracy curves during training will pop up.

![placeholder](/figure/28.png)
Figure 28 CNN Training Process Curves

After closing the training curves, evaluation metrics including “Accuracy,” “Precision,” “Recall,” “F1 Score,” and “ROC-AUC” will be displayed.

![placeholder](/figure/29.png)
Figure 29 CNN Evaluation Metrics

After closing the evaluation metrics, the model will be used for prediction, and results will be loaded directly into the software. All generated figures and the model will be saved in the “Model Save Path.”

![placeholder](/figure/30.png)
Figure 30 CNN Result Map

### 6.1.3 Graph Neural Network

Click [Mineral Prospectivity Mapping]→[Supervised Learning Algorithm]→[Graph Neural Network] to open the GNN model construction Interface (Figure 31).

![placeholder](/figure/31.png)
Figure 31 GNN Model Construction Interface - Model Training

- **Dataset Path:** Select the directory containing dataset generated for supervised learning samples (make sure to check “Graph Construction Distance Threshold”). Choose the file path by clicking ![placeholder](/figure/openfileico.png).
- **Loss Function:** Measures discrepancy between predicted and true values. The software provides cross-entropy loss; no adjustment needed.
- **Epochs:** Number of training iterations. Input type is integer; users can adjust.
- **Learning Rate:** Learning rate for gradient descent optimization. Input type is non-zero decimal; users can adjust.
- **Optimizer:** The optimizer used during training. The software provides Adam; no adjustment needed.
- **Model Save Path:** Path to save the trained model. Select via ![placeholder](/figure/openfileico.png).
- **Model Name:** Filename for the trained model; users can input.

After setting parameters, click [Train] to train and predict using the default network structure, or click [Model Design] to customize the network.
As an example, click [Model Design] to customize the network layers (Figure 32). When designing the network, first set graph convolutional or graph attention layers for spatial feature extraction, then add fully connected layers for classification. The last layer must be a fully connected layer with output dimension 2 and softmax activation. Model parameters refer to explanations in section 5.1.

![placeholder](/figure/32.png)
Figure 32 GNN Model Construction - Model Design

After design, click [Save] to save the model structure, then click [Train] to train and predict using the custom network. A training progress bar will appear. Upon completion, loss and accuracy curves during training will pop up.

![placeholder](/figure/33.png)
Figure 33 GNN Training Process Curves

After closing the training curves, evaluation metrics including “Accuracy,” “Precision,” “Recall,” “F1 Score,” and “ROC-AUC” will be displayed.

![placeholder](/figure/34.png)
Figure 34 GNN Evaluation Metrics

After closing the evaluation metrics, the model will be used for prediction, and results will be loaded directly into the software. All generated figures and the model will be saved in the “Model Save Path”.
![placeholder](/figure/35.png)
Figure 35 GNN Result Map

## 6.2 Self-Supervised Learning Methods

### 6.2.1 Graph Self-Supervised Network

Click [Mineral Prospectivity Mapping]→[Self-Supervised Learning Algorithm]→[Graph Self-Supervised Network] to open the graph self-supervised network construction Interface (Figure 36).
![placeholder](/figure/36.png)
Figure 36 Graph Self-Supervised Network Construction Interface - Model Training

- **Dataset Path:** Select the directory containing dataset generated for supervised learning samples (make sure to check “Graph Construction Distance Threshold”). Choose the file path by clicking ![placeholder](/figure/openfileico.png).
- **Model Save Path:** Path to save the trained model. Select via ![placeholder](/figure/openfileico.png).
- **Model Name:** Filename for the trained model; users can input.
- **Pretraining Phase Hyperparameters**
  - **Epochs:** Number of iterations for self-supervised pretraining. Input type is integer; users can adjust.
  - **Pretraining Learning Rate:** Learning rate during self-supervised pretraining. Input type is non-zero decimal; users can adjust.
  - **Mask Ratio:** Proportion of edges randomly masked when generating mask graphs. Input type is decimal between 0 and 1; users can adjust.
  - **Masked Graph Number:** Number of mask graphs generated. Input type is integer; users can adjust.
- **Fine-tuning Phase Hyperparameters**
  - **Epochs:** Number of iterations during fine-tuning. Input type is integer; users can adjust.
  - **Fine-tuning Model Learning Rate:** Learning rate for the pretrained model during fine-tuning. Input type is non-zero decimal; users can adjust.
  - **Classifier Learning Rate:** Learning rate for the classifier during fine-tuning. Input type is non-zero decimal; users can adjust.

In the self-supervised learning phase, contrastive loss is used; in the fine-tuning phase, cross-entropy loss is used. After setting parameters, click [Train] to train and predict using the default network structure, or click [Model Design] to customize the network.
As an example, click [Model Design] to customize the network layers (Figure 37). When designing the network, first set graph convolutional or graph attention layers for spatial feature extraction (the pretraining model), then add fully connected layers as the classifier. The last layer must be a fully connected layer with output dimension 2 and softmax activation. Model parameters refer to explanations in section 5.1.

![placeholder](/figure/37.png)
Figure 37 Graph Self-Supervised Network Construction Interface - Model Design

If geological constraints need to be added, click  [Geological Constraint] to open the Add Constraint Layer interface, where you can add geological constraint layer and set corresponding loss weights (decimal between 0 and 1).

![placeholder](/figure/38.png)
Figure 38 Adding Geological Constraint Interface

After setting, click [OK] to save, then from the model design interface click [Save] to complete model structure design and add geological constraints. Then click [Train] to train and predict using the custom network. A training progress bar will appear. When progress exceeds 50%, pretraining ends and the pretraining loss graph pops up. After closing it, fine-tuning begins. When all training completes, loss and accuracy curves during training will pop up.

![placeholder](/figure/39.png)
Figure 39 Graph Self-Supervised Network Pretraining Process Curves

![placeholder](/figure/40.png)
Figure 40 Graph Self-Supervised Network Fine-tuning Process Curves

After closing the training curves, evaluation metrics including “Accuracy,” “Precision,” “Recall,” “F1 Score,” and “ROC-AUC” will be displayed.
![placeholder](/figure/41.png)
Figure 41 Graph Self-Supervised Network Evaluation Metrics
After closing the evaluation metrics, the model will be used for prediction, and results will be loaded directly into the software. All generated graphs and the model will be saved in the “Model Save Path.”
![placeholder](/figure/42.png)
Figure 42 Graph Self-Supervised Network Result Map

## 6.3 Reinforcement Learning Methods

### 6.3.1 Graph Reinforcement Learning Algorithm

Click [Mineral Prospectivity Mapping]→[Reinforcement Learning Algorithm]→[Graph Reinforcement Learning Algorithm] to open the graph reinforcement learning model construction window (Figure 43).

![placeholder](/figure/43.png)
Figure 43 Graph Reinforcement Learning Network Construction Interface - Model Training

- **Dataset Path:** Select the directory containing dataset generated for supervised learning samples. Choose the file path by clicking ![placeholder](/figure/openfileico.png).
- **Geochemical Anomaly Map:** Path to the geochemical anomaly map. You can select from layers already loaded in the software via dropdown or choose a file path by clicking ![placeholder](/figure/openfileico.png).
- **Model Save Path:** Path to save the trained model. Select via ![placeholder](/figure/openfileico.png).
- **Model Name:** Filename for the trained model; users can input.
- **Training Hyperparameters Setting**
- **Episodes:** Total number of episode selected during reinforcement learning. Input type is integer; users can adjust.
- **Neighbor Number:** Number of neighbors connected to each node during graph construction. Input type is integer; users can adjust.
- **Learning Rate:** Learning rate for the policy model. Input type is non-zero decimal; users can adjust.
- **Batch Size:** Number of graphs input at once during policy network training. Input type is non-zero decimal; users can adjust.
- **Episode Length:** Length of each episode selected during reinforcement learning. Input type is integer; users can adjust.
- **Batch Number:** Number of times graphs are input during policy network training. Input type is non-zero decimal; users can adjust.
- **Exploration Distance Threshold:** Maximum distance the model explores each time when generating episode during reinforcement learning. Input type is non-zero decimal; users should adjust based on their data.
  
After setting parameters, click [Train] to train and predict using the default network structure, or click [Model Design] to customize the network.
As an example, click [Model Design] to customize the network layers (Figure 44). When designing the network, first set graph convolutional or graph attention layers for spatial feature extraction, then add fully connected layers as the classifier. The last layer must be a fully connected layer with output dimension 2 and softmax activation. Model parameters refer to explanations in section 5.1.

![placeholder](/figure/44.png)
Figure 44 Graph Reinforcement Learning Network Construction Interface - Model Design

If geological constraints need to be added, click [Geological Constraint] to open the geological constraint interface, where you can add geological constraint layer and set corresponding loss weights (decimal between 0 and 1).

![placeholder](/figure/45.png)
Figure 45 Adding Geological Constraint Interface

After setting, click [OK] to save, then from the model Design interface click [Save] to complete model structure design and add geological constraints. Then click [Train] to train and predict using the custom network. A training progress bar will appear. Reinforcement learning training is relatively slow and divided into three stages: generating reinforcement learning data (episode), reinforcement learning, and prediction. The software may become unresponsive during this time; please do not close it. After training completes, the training process curve will be generated (Figure 46).

![placeholder](/figure/46.png)
Figure 46 Graph Reinforcement Learning Network Training Process Curve

After closing the training curve, the model will be used for prediction, and results will be loaded directly into the software. All generated curves and the model will be saved in the “Model Save Path.”

![placeholder](/figure/47.png)
Figure 47 Graph Reinforcement Learning Network Result Map

# 7 Others

## 7.1 Evaluation Curves

Click [Help]→[Evaluation Curve] to open the evaluation curve interface.

![placeholder](/figure/48.png)
Figure 48 Evaluation Curve Interface

- **Positive Points Layer:** Use known mineral deposits as positive points. In this example, Fe_deposits.shp is used as positive points layer. You can add shapefiles already loaded in the software via dropdown or select files by clicking the button.
- **Result Map:** Path to result maps (including geochemical anomaly maps and mineral prospectivity maps). You can select from layers already loaded in the software via dropdown or choose a file path by clicking ![placeholder](/figure/openfileico.png).
- **Result Map Set:** Stores paths of added result maps. This interface will by default load raster data already loaded in the software into this list; users can also add or delete data manually.
- **Save Path:** Path to save evaluation curves. Select via ![placeholder](/figure/openfileico.png).

After setting all parameters, click [OK] to generate evaluation curves: Receiver Operating Characteristic (ROC) curve and Success Rate curve. The curves and generated tabular data will be saved in the specified path.

![placeholder](/figure/49.png)
Figure 49 Evaluation Curves

## 7.2 Creating Geological Constraint Layers

Click [Help]→[Geological Constraint Layer Generation] to open the geological constraint layer generation interface.

![placeholder](/figure/50.png)
Figure 50 Geological Constraint Layer Generation Interface

- **Positive Points Layer:** Use known mineral deposits as positive samples. In this example, Fe_deposits.shp is used as positive sample data. You can add shapefiles already loaded in the software via dropdown or select files by clicking the button.
- **Ore-control Entity:** Path to Ore-control entities. You can select from layers already loaded in the software via dropdown or choose a file path by clicking ![placeholder](/figure/openfileico.png).
- **Processing Area:** The study area boundary and coordinate information, which should be raster data. In this example, any one of the evidence layers of the study area can be used. You can select from raster data already loaded in the software via dropdown or choose a file path by clicking ![placeholder](/figure/openfileico.png).
- **Geological Constraint Layer Output Path:** Path to save the geological constraint layer. Select via ![placeholder](/figure/openfileico.png).
- **Geological Constraint Layer Output Name:** Name of the output layer; users can design as desired.
After setting all parameters, click [OK] to create the geological constraint layer. Upon completion, the power-law fitting curve and constraint layer will be output and saved in the specified paths.

![placeholder](/figure/51.png)
Figure 51 Power-Law Curve

![placeholder](/figure/52.png)
Figure 52 Geological Constraint Layer Display
