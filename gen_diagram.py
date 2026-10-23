"""Generate pipeline diagram using Gemini 2.5 Flash Image."""
import mimetypes
import os
from google import genai
from google.genai import types

PROMPT = """
Create a publication-quality scientific algorithm pipeline diagram for a cross-species gene mapping model. 
The diagram should flow LEFT to RIGHT with 7 numbered steps (①–⑦). Clean white background, flat 2D style, 
thin lines, rounded boxes, sans-serif font. Similar style to computational biology paper figures 
(like SATURN paper by Rosen et al.).

COLOR CODING (strictly follow):
- Mouse/source species elements: ORANGE (#E67E22)
- Human/target species elements: PURPLE (#8E44AD)  
- GO-BP (biological process) terms: GREEN (#27AE60)
- GO-MF (molecular function) terms: BLUE (#2980B9)
- Disease terms: RED (#C0392B)
- Shared/cross-species bridge: GOLD (#F1C40F)

TITLE at top: "Cross-Species Functional Gene Equivalence Mapping"
Subtitle: "Unsupervised many-to-many functional projection via HGT + FGW Optimal Transport"

=== STEP ① DATA INPUT (far left) ===
Two groups of small colored grid matrices (representing sparse gene×term association matrices):

TOP (orange zone, labeled "Mouse genes (24,110)"):
- Green grid "GO-BP" 
- Blue grid "GO-MF"
- Red grid "Disease"
Each grid has scattered colored cells on white background.

BOTTOM (purple zone, labeled "Human genes (31,424)"):
- Same three grids mirrored

Between them, gold dashed lines connecting column headers with label "Shared annotation terms (12,448 BP | 4,609 MF | 22,306 Disease)"

Arrow labeled "TF-IDF + L2 norm" pointing right →

=== STEP ② PER-MODALITY SVD ===
Three stacked rounded boxes, each in its modality color:
- GREEN box: "SVD_BP (d=128), var=37.4%"
- BLUE box: "SVD_MF (d=128), var=59.7%"  
- RED box: "SVD_Disease (d=128), var=41.0%"

Below them, a GRAY box: "Unified SVD → gene initialization"

Each outputs a narrow dense matrix (gene embeddings per modality).
Arrow right →

=== STEP ③ HETEROGENEOUS GRAPH + HGT (visual centerpiece) ===
A bipartite graph structure:
- TOP ROW: 6 ORANGE circles = Mouse gene nodes
- BOTTOM ROW: 6 PURPLE circles = Human gene nodes
- MIDDLE ROW: 3 larger circles with GOLD glow border = Shared term nodes:
  - GREEN circle "BP"
  - BLUE circle "MF"  
  - RED circle "Dis"

Semi-transparent edges: orange lines from mouse genes to shared terms, purple lines from human genes to shared terms. NO direct mouse-human edges.

Label the shared nodes: "Functional Bridge — no ortholog supervision needed"

Below: box "HGT: 2-layer, 4 heads, skip gate α≈0.8"

Callout: "Genes sharing similar functions → similar embeddings via shared term neighbors"

Arrow right →

=== STEP ④ PER-MODALITY FUSION ===
Three parallel streams converging:
- GREEN stream: "z_BP = Aggregate + Z_BP residual"
- BLUE stream: "z_MF = Aggregate + Z_MF residual"
- RED stream: "z_Dis = Aggregate + Z_Dis residual"

All three converge into a PURPLE diamond: "Attention Fusion: E = Σαᵢzᵢ"

Small inset: show 3 colored arrows pointing in DIFFERENT directions (representing genuinely different modality embeddings)

Output: "E_fused (d=128)"

Arrow right →

=== STEP ⑤ CANDIDATE SELECTION ===
Box: "SVD Cosine Retrieval → Top-128 candidates"
Below: "MMR Diversity Filter (λ=0.25)"

Visual: one large ORANGE circle (query gene) connected by lines to a cloud of ~10 small PURPLE circles (candidate genes). Some brighter, some dimmer.

Arrow right →

=== STEP ⑥ FGW OPTIMAL TRANSPORT (mathematical core) ===
Two inputs feeding into solver:
- Left: "Structural Cost (α=0.5): ‖Eᵢ−Eⱼ‖²" with small embedding space diagram
- Right: "Feature Cost (1−α=0.5): annotation overlap" with small vector comparison diagram

Central box: "KL-Unbalanced Sinkhorn, ε=0.05, τ=0.05/0.5"

Output: A small 5×5 HEATMAP (red color scale) = coupling matrix T[i,j]
- NOT diagonal — show off-diagonal elements (many-to-many)
- Label: "T[i,j] = functional correspondence strength"

Side note: "Unbalanced OT: genes without equivalents get low coupling (mass stays)"

Arrow right →

=== STEP ⑦ OUTPUT (far right) ===
Example result card:

Query (orange box): "Trp53 (mouse)" with small mouse icon

Results (purple):
- Bar chart: TP53 (0.89), TP73 (0.35), TP63 (0.28)
- Label: "Many-to-many: 1 mouse gene → 3 human functional equivalents"

Below:
- "Related GO: apoptosis, DNA repair"
- "Related Disease: Li-Fraumeni syndrome"

Bidirectional arrow at bottom: "↔ Mouse→Human AND Human→Mouse"

=== BOTTOM BANNER ===
Four key principles with colored dots:
① Orange: "Purely Unsupervised — no ortholog labels"
② Purple: "Many-to-Many Mapping — not 1:1 ortholog"  
③ Gold: "Per-modality Fusion — different BP/MF/Disease signals"
④ Green: "Bidirectional Training — symmetric mapping"
"""

def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=PROMPT)],
        ),
    ]
    
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )
    
    out_dir = "/Genomics/chanlab/Ze/MGPmodel/Project1_DiseaseGo2Gene"
    file_index = 0
    
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=config,
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                fname = f"{out_dir}/pipeline_diagram_{file_index}{ext}"
                with open(fname, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"Saved: {fname}")
                file_index += 1
            elif part.text:
                print(part.text)
    
    if file_index == 0:
        print("WARNING: No image generated!")

if __name__ == "__main__":
    main()
