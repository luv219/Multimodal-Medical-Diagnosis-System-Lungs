# Multimodal Medical Diagnosis System — UML Diagrams

Below are the requested UML and Architecture diagrams generated using Mermaid syntax. You can include these directly in your Markdown documents, or render them to PNG/SVG using a Mermaid live editor or IDE plugin for your IEEE report.

## 1. System Component Architecture
This diagram illustrates the high-level components and data flow of the system.

```mermaid
graph TD
    subgraph Data Pipeline
        A[NIH ChestX-ray14 Dataset] -->|Images & Labels| B[NIHChestDataset]
        B -->|Image Tensors, Labels, Filenames| C[DataLoader Factory]
        C --> D(Train DataLoader)
        C --> E(Val DataLoader)
        C --> F(Test DataLoader)
    end

    subgraph Modeling & Training
        D --> G[Training Script]
        E --> G
        G -->|Model Initialization| H{Model Registry}
        H -.->|get_densenet| I[DenseNet-121]
        H -.->|get_swin| J[Swin Transformer]
        H -.->|get_hybrid| K[CNN-Transformer Hybrid]
        G -->|Optimize| L[AUC-Margin Loss / PESG]
        L -->|Checkpoints| M[(Saved Models)]
    end

    subgraph Evaluation & Explainability
        F --> N[Evaluation Script]
        M --> N
        N -->|Metrics| O[JSON Reports]
        M --> P[Explainability Script]
        P -->|GradCAM++| Q[Saliency Maps]
        O --> R[Dashboard]
    end
```

## 2. Class Diagram
This diagram outlines the core Python classes and their relationships.

```mermaid
classDiagram
    class NIHChestDataset {
        +String split
        +Dict image_lookup
        +List filenames
        +List labels
        +Tensor class_weights
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class DenseNet121Classifier {
        +densenet Module
        +classifier Module
        +freeze_backbone(freeze: bool)
        +get_cam_target_layer() Module
        +forward(x: Tensor) Tensor
        +model_info()
    }
    
    class SwinClassifier {
        +swin Module
        +classifier Module
        +freeze_backbone(freeze: bool)
        +get_cam_target_layer() Module
        +forward(x: Tensor) Tensor
        +model_info()
    }
    
    class HybridClassifier {
        +cnn_backbone Module
        +projection Conv2d
        +pos_encoding Parameter
        +transformer TransformerEncoder
        +classifier Module
        +freeze_backbone(freeze: bool)
        +get_cam_target_layer() Module
        +forward(x: Tensor) Tensor
        +model_info()
    }
    
    class AUCMLoss {
        +forward(preds, targets) Tensor
    }
    
    class FocalLoss {
        +forward(preds, targets) Tensor
    }
    
    NIHChestDataset <.. DataLoaderFactory : instantiates
    DenseNet121Classifier ..|> ModelRegistry : registered
    SwinClassifier ..|> ModelRegistry : registered
    HybridClassifier ..|> ModelRegistry : registered
```

## 3. Hybrid Model Architecture (Activity/Flow)
Since the CNN-Transformer Hybrid is a custom architecture, this sequence/flow diagram details how a single tensor passes through it.

```mermaid
sequenceDiagram
    participant Input as Input Image<br/>(B, 3, 224, 224)
    participant ResNet as ResNet-50 Layer 4
    participant Conv as 1x1 Conv Projection
    participant Pos as Positional Encoding
    participant Trans as Transformer Encoder
    participant Pool as Global Average Pool
    participant Head as Classification Head
    
    Input->>ResNet: Forward Pass
    ResNet-->>Conv: Feature Map (B, 2048, 7, 7)
    Conv-->>Pos: Projected (B, 256, 7, 7)
    Note over Pos: Flatten spatial dims to 49 tokens
    Pos-->>Trans: + Pos Encoding (B, 49, 256)
    Trans-->>Pool: Contextualized Tokens (B, 49, 256)
    Pool-->>Head: Sequence Mean (B, 256)
    Head-->>Input: Raw Logits (B, 14)
```

## 4. Training Sequence Diagram
This diagram shows the interactions during the training phase.

```mermaid
sequenceDiagram
    participant User
    participant Config as config.py
    participant Train as train.py
    participant Data as dataset.py
    participant Models as models/__init__.py
    
    User->>Train: python train.py --model hybrid --loss aucm
    Train->>Config: get runtime config & paths
    Train->>Data: get_dataloaders()
    Data-->>Train: train_loader, val_loader
    Train->>Models: get_model("hybrid")
    Models-->>Train: HybridClassifier instance
    
    loop Warmup Phase (Epochs 1-3)
        Train->>Models: model.freeze_backbone(True)
        Train->>Train: Train classification head only
    end
    
    loop Fine-tuning Phase (Epochs 4+)
        Train->>Models: model.freeze_backbone(False)
        Train->>Train: Train end-to-end (PESG Optimizer)
        Train->>Train: Validate on val_loader
        alt If Macro AUC improves
            Train->>Train: Save checkpoints/best.pth
        end
    end
    
    Train-->>User: Training complete
```
