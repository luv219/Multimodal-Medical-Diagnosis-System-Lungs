# Multimodal Medical Diagnosis System — Functional Diagrams

Below are the functional diagrams modeled using Mermaid syntax. These describe the functional blocks and data flow of the system, focusing on what the system *does* rather than its internal software class structure.

## 1. Overall System Functional Block Diagram (Level 0 DFD)
This diagram provides a high-level view of the primary functional modules of the diagnostic system and the data exchanged between them.

```mermaid
graph LR
    %% External Entities
    Dataset[NIH ChestX-ray14<br/>Database]
    User[Clinician / Researcher]
    
    %% Functional Processes
    subgraph Diagnostic System
        DataPrep[1. Data Ingestion &<br/>Preprocessing]
        ModelTrain[2. Disease Classification<br/>Training]
        Infer[3. Multi-label<br/>Inference]
        Explain[4. Visual<br/>Explainability]
        Reporting[5. Dashboard &<br/>Analytics]
    end
    
    %% Data Flow
    Dataset -->|Raw X-Rays & Meta-labels| DataPrep
    DataPrep -->|Normalized Tensors| ModelTrain
    ModelTrain -->|Trained Weights| Infer
    ModelTrain -->|Trained Weights| Explain
    
    User -->|New X-Ray| Infer
    Infer -->|Disease Probabilities| Reporting
    Infer -->|Target Class| Explain
    Explain -->|GradCAM++ Saliency Maps| Reporting
    
    Reporting -->|Visual UI & Metrics| User
```

## 2. Training Optimization Flow (Functional View)
This diagram breaks down the training function, specifically highlighting the AUC-Margin loss and PESG optimizer workflow.

```mermaid
graph TD
    Input[Batched Images & Labels] --> Fwd[Model Forward Pass]
    
    subgraph Forward Pass 
        Fwd -->|Logits| LossFunc[Calculate AUC-Margin Loss]
        LossFunc -.->|Imratio Info| ClassWeights[Class Imbalance Adjustment]
    end
    
    subgraph Backward Pass
        LossFunc -->|Gradients| PESG[PESG Optimizer Update]
        PESG -->|Update Params| ModelWeights[(Model Weights)]
    end
    
    ModelWeights -->|Validation Eval| Val[Calculate Macro AUC]
    Val --> Cond{Is Best AUC?}
    Cond -->|Yes| Save[Update best.pth]
    Cond -->|No| Next[Next Epoch]
    Save --> Next
```

## 3. Explainability Functional Flow (GradCAM++)
This diagram illustrates the functional sequence to generate explainable visual maps from an input image.

```mermaid
graph TD
    Img[Original Chest X-Ray] --> Pre[Preprocess<br/>Resize & Normalize]
    Pre -->|Tensor| FwdPass[Model Inference]
    
    FwdPass -->|Logits| TopK[Extract Top-K Diseases]
    TopK -->|Target Disease Index| Grad[GradCAM++ Function]
    
    %% Transformer vs CNN handling
    FwdPass -.->|Layer Outputs| FeatMap{Feature Map Type}
    FeatMap -->|4D Tensor| CNN[CNN Pipeline]
    FeatMap -->|3D Sequence| Trans[Transformer Pipeline]
    Trans -->|Reshape & Transpose| CNN
    
    CNN -->|Spatial Gradients| Grad
    
    Grad -->|Raw Heatmap| Up[Upsample to 224x224]
    Up -->|Color Mapped| Overlay[Overlay on Original Image]
    Overlay --> Output[Final Explanatory Figure]
```

## 4. Evaluation and Dashboard Data Flow
This diagram shows how evaluation data is generated and transformed into dashboard visualizations.

```mermaid
flowchart LR
    subgraph Evaluation
        Model[(Trained Models)] --> EvalRun[Run evaluate.py]
        TestSet[(Test Dataset)] --> EvalRun
        EvalRun -->|Find Optimal Cutoffs| F1[Maximize F1 per Disease]
        F1 --> JSON[(Results JSON)]
    end

    subgraph Dashboard Generation
        JSON --> Read[Parse Metrics]
        Read --> HM_AUC[Generate AUC Heatmap]
        Read --> HM_F1[Generate F1 Heatmap]
        Read --> Bar[Generate Macro Bar Charts]
        Read --> Table[Generate Best-Model Table]
        
        HM_AUC --> Render[Render 2x2 Panel]
        HM_F1 --> Render
        Bar --> Render
        Table --> Render
        Render --> PNG[dashboard.png]
    end
```
