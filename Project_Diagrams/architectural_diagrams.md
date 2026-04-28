# Multimodal Medical Diagnosis System — Architectural Diagrams

Below are the architectural diagrams detailing the specific neural network structures used in your system. These are particularly useful for the Methodology or Architecture section of an IEEE paper, as they show layer transformations and tensor dimensions.

## 1. DenseNet-121 Classification Architecture
This diagram outlines the architecture of the DenseNet-121 baseline model, including the custom classification head.

```mermaid
graph TD
    Input["Input X-Ray Image<br/>(B, 3, 224, 224)"] --> Backbone
    
    subgraph DenseNet-121 Backbone
        Backbone["DenseNet-121 (ImageNet Pretrained)"]
        Backbone --> Features["denseblock4 Features<br/>(B, 1024, 7, 7)"]
        Features --> GAP["Global Average Pooling<br/>(B, 1024)"]
    end
    
    subgraph Classification Head
        GAP --> L1["Linear Projection<br/>(1024 → 512)"]
        L1 --> Relu["ReLU Activation"]
        Relu --> L2["Linear Classifier<br/>(512 → 14)"]
    end
    
    L2 --> Output["Raw Logits Output<br/>(B, 14)"]
    
    %% Styling
    classDef tensor fill:#f9f,stroke:#333,stroke-width:2px;
    class Input,Features,GAP,Output tensor;
```

## 2. Swin-Tiny Transformer Architecture
This diagram illustrates the Swin Transformer model pipeline, emphasizing the layer normalization and GELU activations in the head.

```mermaid
graph TD
    Input["Input X-Ray Image<br/>(B, 3, 224, 224)"] --> Patch
    
    subgraph Swin-Tiny Backbone
        Patch["Patch Partition & Linear Embedding"] --> Blocks["Swin Transformer Blocks<br/>(Window Attention + Shifted Window)"]
        Blocks --> Features["layers[-1] Outputs<br/>(B, 49, 768)"]
        Features --> Pool["Pooling / Flattening<br/>(B, 768)"]
    end
    
    subgraph Classification Head
        Pool --> LN["LayerNorm(768)"]
        LN --> L1["Linear Projection<br/>(768 → 512)"]
        L1 --> Gelu["GELU Activation"]
        Gelu --> L2["Linear Classifier<br/>(512 → 14)"]
    end
    
    L2 --> Output["Raw Logits Output<br/>(B, 14)"]
```

## 3. CNN-Transformer Hybrid Architecture
This is a detailed architectural view of your custom Hybrid model, showing how spatial features from a CNN are converted into a sequence for the Transformer. This is often the most critical diagram for a paper introducing a novel hybrid approach.

```mermaid
graph TD
    Input["Input X-Ray Image<br/>(B, 3, 224, 224)"] --> ResNet
    
    subgraph CNN Feature Extractor
        ResNet["ResNet-50 Backbone<br/>(Pretrained)"] --> Layer4["layer4 Spatial Map<br/>(B, 2048, 7, 7)"]
        Layer4 --> Conv1x1["1x1 Conv Projection<br/>(B, 256, 7, 7)"]
    end
    
    subgraph Sequence Preparation
        Conv1x1 --> Flatten["Flatten Spatial Dims<br/>(B, 256, 49)"]
        Flatten --> Transpose["Transpose<br/>(B, 49, 256)"]
        Transpose --> PosEnc["+ Learnable Positional Encoding<br/>(1, 49, 256)"]
    end
    
    subgraph Transformer Encoder
        PosEnc --> TE1["Transformer Layer 1<br/>(8 Heads, d_model=256)"]
        TE1 --> TE2["Transformer Layer 2<br/>(8 Heads, d_model=256)"]
        TE2 --> ContextTokens["Contextualized Tokens<br/>(B, 49, 256)"]
    end
    
    subgraph Global Pooling & Classifier
        ContextTokens --> SeqMean["Sequence Mean (Dim 1)<br/>(B, 256)"]
        SeqMean --> LN["LayerNorm(256)"]
        LN --> LinearHead["Linear Classifier<br/>(256 → 14)"]
    end
    
    LinearHead --> Output["Raw Logits Output<br/>(B, 14)"]
```

## 4. Overall Multimodal / Multi-label Architecture Pipeline
This diagram abstracts the network architectures to show the overall modeling pipeline from input to the AUC-Margin loss function during training.

```mermaid
flowchart LR
    Img["Input Image<br/>(3x224x224)"] --> Model{Architecture Selection}
    
    Model -->|DenseNet-121| F1["CNN Features"]
    Model -->|Swin-Tiny| F2["Transformer Tokens"]
    Model -->|Hybrid| F3["CNN + Transformer"]
    
    F1 --> Head["Classification Head"]
    F2 --> Head
    F3 --> Head
    
    Head --> Logits["Raw Logits (14 classes)"]
    
    Labels["Ground Truth Multi-hot Vector<br/>(14 classes)"] --> Loss
    Logits --> Loss["libauc AUCM_MultiLabel Loss"]
    
    Loss -->|Imratio Config| PESG["PESG Optimizer"]
    PESG -->|Backpropagation| Model
```
