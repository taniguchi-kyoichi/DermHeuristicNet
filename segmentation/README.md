1. **Introduction**
- 皮膚科医のABCDルールに基づく診断知識を深層学習に組み込むことで、より信頼性の高い診断支援システムを目指す

2. **Related Work**
- 従来の深層学習による皮膚病変診断は特徴抽出がブラックボックス化しており、医師の診断プロセスとの対応が不明確

3. **Proposed Method**
- セグメンテーションによる病変の境界情報と、原画像からの視覚的特徴を組み合わせるデュアルストリームアーキテクチャを提案

4. **Experiments**
- HAM10000データセットを用いて、提案手法と従来手法の性能比較および各モジュールの有効性を検証

5. **Results and Discussion**
- 提案手法は従来手法と比較して診断精度が向上し、特に境界の不規則性が重要な診断指標となる病変タイプで顕著な改善を示した

6. **Conclusion**
- 医学的知見を明示的にモデルに組み込むアプローチの有効性を実証し、より解釈可能な診断支援システムの実現可能性を示した

``` mermaid
flowchart LR
    subgraph Input
        img[Original Image]
    end

    subgraph Segmentation Stream
        seg[Lesion Segmentation Module]
        seg_feat[Efficient Feature Extractor]
    end

    subgraph Main Stream
        irv2[InceptionResNetV2]
    end

    subgraph Fusion
        ff[Feature Fusion Module]
        style ff fill:#000,stroke:#333,stroke-width:4px
    end

    subgraph Output
        cls[Classification]
    end

    img --> seg
    img --> irv2
    seg --> seg_feat
    seg_feat --> ff
    irv2 --> ff
    ff --> cls

    %% Segmentation Stream Details
    subgraph seg_details[Segmentation Module Detail]
        direction TB
        preproc[Preprocessing]
        thresh[Adaptive Thresholding]
        morph[Morphological Operations]
        refine[Boundary Refinement]
        preproc --> thresh
        thresh --> morph
        morph --> refine
    end

    %% Feature Fusion Details
    subgraph fusion_details[Feature Fusion Detail]
        direction TB
        channel[Channel Attention]
        spatial[Spatial Attention]
        combine[Feature Integration]
        channel --> combine
        spatial --> combine
    end
```