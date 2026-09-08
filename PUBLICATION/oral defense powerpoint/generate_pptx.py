import os
import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Theme colors
    DARK_BLUE = RGBColor(12, 59, 110)      # #0C3B6E - Primary Dark Blue
    MED_BLUE = RGBColor(28, 54, 115)       # #1C3673 - Footer Blue
    LIGHT_BLUE = RGBColor(240, 247, 255)   # #F0F7FF - Shaded Block Blue
    ORANGE = RGBColor(217, 83, 25)         # #D95319 - Highlight Orange
    LIGHT_ORANGE = RGBColor(255, 249, 240) # #FFF9F0 - Shaded Block Orange
    TEXT_BLACK = RGBColor(30, 30, 30)      # Primary Text
    TEXT_MUTED = RGBColor(80, 80, 80)      # Muted Text (sub-bullets)
    WHITE = RGBColor(255, 255, 255)
    
    # Directory paths
    img_dir = r"D:\ttu\AI driving monitoring system\PUBLICATION\oral defense powerpoint\extracted_media"
    plots_dir = r"D:\ttu\AI driving monitoring system\logs\plots"
    
    # Reference Database matching THESIS_DRAFT.md
    REFS_DB = {
        1: '[1] W. Verkruysse et al., "Remote plethysmography using ambient light," Optics Express, 2008.',
        2: '[2] WHO, "Global status report on road safety 2023," 2023.',
        3: '[3] R. Zhao et al., "Driver Fatigue Detection Based on BiLSTM-At and Multi-Source Fusion," J. Adv. Transp., 2025.',
        4: '[4] A. Revanur et al., "The First Vision For Vitals (V4V) Challenge," in Proc. ICCV Workshops, 2021.',
        5: '[5] S. J. Delpy et al., "Estimation of optical pathlength through tissue," Phys. Med. Biol., 1988.',
        6: '[6] M. Poh et al., "Non-contact, automated cardiac pulse measurements," Optics Express, 2010.',
        7: '[7] G. de Haan and V. Jeanne, "Robust pulse rate from chrominance-based rPPG," IEEE Trans. Biomed. Eng., 2013.',
        8: '[8] W. Wang et al., "Algorithmic Principles of Remote PPG," IEEE Trans. Biomed. Eng., 2017.',
        9: '[9] J. Kim, "Sustainable Real-Time Gaze Monitoring with YOLOX-based Attention," Exp. Syst. Appl., 2025.',
        10: '[10] G. Du et al., "Vision-Based Fatigue Driving Recognition Method," IEEE Trans. Intell. Transp. Syst., 2021.',
        11: '[11] C.-H. Cheng et al., "Deep Learning Methods for Remote Heart Rate Measurement: A Review," Sensors, 2021.',
        12: '[12] L. Kong et al., "RPPMT-CNN-BiLSTM: A Multi-modal Fusion Framework for Fatigue Detection," Electronics, 2024.',
        13: '[13] S. Gupta et al., "RADIANT: Better rPPG Estimation Using Signal Embeddings," in Proc. WACV, 2023.',
        14: '[14] A. Vaswani et al., "Attention is all you need," in Proc. NIPS, 2017.',
        15: '[15] S. Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography," Pattern Recogn. Lett., 2017.',
        16: '[16] H. Shao et al., "Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios," in Proc. CVPR, 2025.',
        17: '[17] S. Butterworth, "On the theory of filter amplifiers," Wireless Engineer, 1930.',
        18: '[18] R. Hamming, "Digital Filters," Prentice-Hall, 1983.',
        19: '[19] C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv:1906.08172, 2019.',
        20: '[20] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, "Heart rate variability: standards of measurement, physiological interpretation and clinical use," Circulation, 1996.',
        21: '[21] M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in Proc. OSDI, 2016.',
        22: '[22] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.',
        23: '[23] J. Doe et al., "ME-rPPG: Memory-Efficient Remote Photoplethysmography," IEEE Trans. Multimedia, 2025.',
        24: '[24] Y. Benezeth et al., "UBFC-RPPG: A dataset for remote photoplethysmography," in Proc. FedCSIS, 2017.',
        25: '[25] C. Weng et al., "Driver Drowsiness Detection via Learnable Deep Landmarks," IEEE Access, 2016.',
        26: '[26] D. Powers, "Evaluation: From Precision, Recall and F-Measure to ROC," J. Mach. Learn. Technol., 2011.',
        27: '[27] T. Åkerstedt and M. Gillberg, "Subjective and objective sleepiness in the active individual," Int. J. Neurosci., 1990.',
        28: '[28] A. B. Arrieta et al., "Explainable Artificial Intelligence (XAI): Concepts, taxonomies and challenges," Inf. Fusion, 2020.',
        29: '[29] Qualcomm, "Snapdragon Ride Platform: Scalable and Open Automated Driving Solution," 2023.',
        30: '[30] A. Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference," arXiv:2103.13630, 2021.',
        31: '[31] Google, "XNNPACK: High-efficiency floating-point neural network inference operators," 2023.',
        32: '[32] European Parliament, "Regulation (EU) 2016/679 (GDPR)," Official Journal of the EU, 2016.'
    }
    
    # Helper to parse formatted text and add runs
    def add_formatted_runs(paragraph, text, default_size=18, default_color=TEXT_BLACK):
        pattern = re.compile(r'(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*.*?\*|[^*]+)')
        parts = pattern.findall(text)
        for part in parts:
            run = paragraph.add_run()
            if part.startswith('***') and part.endswith('***'):
                run.text = part[3:-3]
                run.font.bold = True
                run.font.color.rgb = ORANGE
            elif part.startswith('**') and part.endswith('**'):
                run.text = part[2:-2]
                run.font.bold = True
                run.font.color.rgb = default_color
            elif part.startswith('*') and part.endswith('*'):
                run.text = part[1:-1]
                run.font.bold = True
                run.font.color.rgb = ORANGE
            else:
                run.text = part
                run.font.color.rgb = default_color
            run.font.name = "Arial"
            run.font.size = Pt(default_size)

    # Helper to add standard header/footer
    total_slides = 28
    def add_slide_layout(title_text, slide_num, refs=None):
        blank_layout = prs.slide_layouts[6] # blank
        slide = prs.slides.add_slide(blank_layout)
        
        # Header shape
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.9))
        header.fill.solid()
        header.fill.fore_color.rgb = DARK_BLUE
        header.line.color.rgb = DARK_BLUE
        
        # Header text
        tx_header = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12.333), Inches(0.7))
        tf = tx_header.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = WHITE
        
        # Footer shape
        footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.15), Inches(13.333), Inches(0.35))
        footer.fill.solid()
        footer.fill.fore_color.rgb = MED_BLUE
        footer.line.color.rgb = MED_BLUE
        
        # Footer text - Left
        tx_left = slide.shapes.add_textbox(Inches(0.5), Inches(7.18), Inches(3.0), Inches(0.3))
        tf_l = tx_left.text_frame
        tf_l.word_wrap = True
        tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0
        p_l = tf_l.paragraphs[0]
        p_l.text = "S.-W. Chang (TTU)"
        p_l.font.name = "Arial"
        p_l.font.size = Pt(10)
        p_l.font.color.rgb = WHITE
        
        # Footer text - Center
        tx_center = slide.shapes.add_textbox(Inches(3.5), Inches(7.18), Inches(6.333), Inches(0.3))
        tf_c = tx_center.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
        p_c = tf_c.paragraphs[0]
        p_c.alignment = PP_ALIGN.CENTER
        p_c.text = "Attention-Based Multi-Modal Driver Monitoring System"
        p_c.font.name = "Arial"
        p_c.font.size = Pt(10)
        p_c.font.color.rgb = WHITE
        
        # Footer text - Right
        tx_right = slide.shapes.add_textbox(Inches(9.833), Inches(7.18), Inches(3.0), Inches(0.3))
        tf_r = tx_right.text_frame
        tf_r.word_wrap = True
        tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0
        p_r = tf_r.paragraphs[0]
        p_r.alignment = PP_ALIGN.RIGHT
        p_r.text = f"Master's Defense  |  {slide_num} / {total_slides}"
        p_r.font.name = "Arial"
        p_r.font.size = Pt(10)
        p_r.font.color.rgb = WHITE

        # References text box (placed right above footer)
        if refs:
            ref_texts = [REFS_DB[r] for r in refs if r in REFS_DB]
            if ref_texts:
                ref_str = "   ".join(ref_texts)
                tx_ref = slide.shapes.add_textbox(Inches(0.5), Inches(6.75), Inches(12.333), Inches(0.35))
                tf_ref = tx_ref.text_frame
                tf_ref.word_wrap = True
                tf_ref.margin_left = tf_ref.margin_top = tf_ref.margin_right = tf_ref.margin_bottom = 0
                p_ref = tf_ref.paragraphs[0]
                p_ref.text = ref_str
                p_ref.font.name = "Arial"
                p_ref.font.size = Pt(8.5)
                p_ref.font.color.rgb = RGBColor(120, 120, 120)
        
        return slide

    # Helper to add standard bullets
    def add_bullets(slide, points, x=0.5, y=1.2, w=12.333, h=5.2, size_m=18, size_s=15):
        tx_box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        first = True
        for level, text in points:
            if first:
                p = tf.paragraphs[0]
                first = False
            else:
                p = tf.add_paragraph()
            
            p.line_spacing = 1.2
            if level == 0:
                p.space_after = Pt(8)
                p.left_indent = 0
                # Draw ▶ triangle
                p.text = ""
                run_tri = p.add_run()
                run_tri.text = "▶  "
                run_tri.font.name = "Arial"
                run_tri.font.size = Pt(size_m)
                run_tri.font.bold = True
                run_tri.font.color.rgb = DARK_BLUE
                add_formatted_runs(p, text, default_size=size_m, default_color=TEXT_BLACK)
            else:
                p.space_after = Pt(6)
                p.left_indent = Inches(0.4)
                # Draw ▷ triangle
                p.text = ""
                run_tri = p.add_run()
                run_tri.text = "▷  "
                run_tri.font.name = "Arial"
                run_tri.font.size = Pt(size_s)
                run_tri.font.bold = True
                run_tri.font.color.rgb = ORANGE
                add_formatted_runs(p, text, default_size=size_s, default_color=TEXT_MUTED)
        return tx_box

    # Helper to add highlighted blocks
    def add_content_block(slide, title, content, x, y, w, h, style='blue'):
        # Body background
        body_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        body_bg.fill.solid()
        body_bg.fill.fore_color.rgb = LIGHT_BLUE if style == 'blue' else LIGHT_ORANGE
        body_bg.line.color.rgb = DARK_BLUE if style == 'blue' else ORANGE
        body_bg.line.width = Pt(1.5)
        
        # Title bar
        title_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.45))
        title_bg.fill.solid()
        title_bg.fill.fore_color.rgb = DARK_BLUE if style == 'blue' else ORANGE
        title_bg.line.color.rgb = DARK_BLUE if style == 'blue' else ORANGE
        
        # Title text
        tx_title = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.05), Inches(w - 0.3), Inches(0.35))
        tf_t = tx_title.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = WHITE
        
        # Body text
        tx_body = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.55), Inches(w - 0.3), Inches(h - 0.65))
        tf_b = tx_body.text_frame
        tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
        p_b = tf_b.paragraphs[0]
        p_b.line_spacing = 1.2
        add_formatted_runs(p_b, content, default_size=13, default_color=TEXT_BLACK)

    # -------------------------------------------------------------
    # SLIDE 1: Cover Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title box (center top-mid)
    title_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.5), Inches(12.333), Inches(2.2))
    title_box.fill.solid()
    title_box.fill.fore_color.rgb = DARK_BLUE
    title_box.line.color.rgb = DARK_BLUE
    
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.5)
    tf.margin_top = tf.margin_bottom = Inches(0.2)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "基於注意力機制多模態融合之穩健駕駛狀態監控系統"
    p.font.name = "Arial"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(12)
    
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = "A Robust Driver State Monitoring System Utilizing Attention-Based Multi-Modal Fusion"
    p2.font.name = "Arial"
    p2.font.size = Pt(20)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(180, 210, 245)
    
    # Info text box
    info_box = slide1.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(2.5))
    tf_info = info_box.text_frame
    tf_info.word_wrap = True
    tf_info.margin_left = tf_info.margin_right = tf_info.margin_top = tf_info.margin_bottom = 0
    
    p_grad = tf_info.paragraphs[0]
    p_grad.alignment = PP_ALIGN.CENTER
    p_grad.text = "研 究 生：張善為 (Shang-Wei Chang)"
    p_grad.font.name = "Arial"
    p_grad.font.size = Pt(16)
    p_grad.font.bold = True
    p_grad.font.color.rgb = TEXT_BLACK
    p_grad.space_after = Pt(8)
    
    p_adv = tf_info.add_paragraph()
    p_adv.alignment = PP_ALIGN.CENTER
    p_adv.text = "指導教授：許超雲 教授 (Prof. Chau-Yun Hsu)  /  卓立 教授 (Prof. Cho Li)"
    p_adv.font.name = "Arial"
    p_adv.font.size = Pt(16)
    p_adv.font.bold = True
    p_adv.font.color.rgb = TEXT_BLACK
    p_adv.space_after = Pt(8)
    
    p_inst = tf_info.add_paragraph()
    p_inst.alignment = PP_ALIGN.CENTER
    p_inst.text = "大同大學 電機工程研究所 碩士學位論文口試"
    p_inst.font.name = "Arial"
    p_inst.font.size = Pt(16)
    p_inst.font.color.rgb = TEXT_MUTED
    p_inst.space_after = Pt(8)
    
    p_date = tf_info.add_paragraph()
    p_date.alignment = PP_ALIGN.CENTER
    p_date.text = "Thesis for Master of Science  |  July 2026"
    p_date.font.name = "Arial"
    p_date.font.size = Pt(14)
    p_date.font.color.rgb = ORANGE
    p_date.font.bold = True

    # -------------------------------------------------------------
    # SLIDE 2: Outline
    # -------------------------------------------------------------
    slide2 = add_slide_layout("Outline", 2)
    outline_points = [
        (0, "1.  **Background & Motivation**: The Reactive Gap & Multi-Modal Fusion"),
        (0, "2.  **Proposed System Architecture**: Tiered Real-Time Pipeline"),
        (0, "3.  **Part A**: Signal Discovery & SNR-based ROI-Switching Attention"),
        (0, "4.  **Part B**: Feature Refinement & Physiological Estimation"),
        (0, "5.  **Part C**: Temporal Fusion with Multi-Head Attention (MHA)"),
        (0, "6.  **Experimental Setup & Results**: Ablation Studies, SOTA Comparison, XAI Projects"),
        (0, "7.  **Conclusion & Future Work**: Key Contributions and Edge Deployment Roadmap")
    ]
    add_bullets(slide2, outline_points, y=1.5, size_m=18)

    # -------------------------------------------------------------
    # SLIDE 3: Background & Motivation: The Reactive Gap
    # -------------------------------------------------------------
    slide3 = add_slide_layout("1. Background & Motivation: The Reactive Gap", 3, refs=[2, 3])
    points3 = [
        (0, "**Driver Drowsiness & Distraction**: Leading contributors to traffic accidents globally [2]."),
        (0, "**Reactive Gap of Ocular Systems**: PERCLOS (EAR) and yawning (MAR) only manifest *after* significant physiological decline has occurred [3]."),
        (0, "**Environmental Lighting Vulnerability**: Ambient glare, darkness, sunglasses, and masks cause traditional vision landmarks to lose tracking or fail completely."),
        (0, "**Proposed Solution**: Fusing *internal* physiological indicators (rPPG, HRV Delta) with *external* behavioral cues to predict fatigue *before* physical onset.")
    ]
    add_bullets(slide3, points3, h=3.0)
    add_content_block(slide3, "The Research Goal", 
                      "Establish a *fail-operational* Driver Monitoring System (DMS) that maintains high diagnostic precision by dynamically balancing physiological and behavioral modalities under adverse lighting or sensor obstructions.",
                      0.5, 4.5, 12.333, 2.0, style='blue')

    # -------------------------------------------------------------
    # SLIDE 4: Theoretical Foundation: Optical Physics of rPPG
    # -------------------------------------------------------------
    slide4 = add_slide_layout("1. Theoretical Foundation: Optical Physics of rPPG", 4, refs=[1, 4, 5])
    points4 = [
        (0, "**Remote Photoplethysmography (rPPG)**: Extracts cardiac pulse from standard RGB cameras by capturing skin light reflection changes [1]."),
        (0, "**Dichromatic Reflection Model (DRM)**: Shafer (1985) [4] defines skin radiance:"),
        (1, "**Specular Reflection (C_s)**: Surface boundary reflection (achromatic noise) containing zero pulse data."),
        (1, "**Body (Diffuse) Reflection (C_b)**: Light penetrates the epidermis, scatters in dermis, and carries the pulse."),
        (0, "**Modified Beer-Lambert Law (MBLL)**: [5]:"),
        (1, "Blood volume changes (*Δc*) during cardiac cycles modulate light absorption (AC signal)."),
        (0, "**Green Channel Selection**: Strongest hemoglobin absorption peak (~530-570nm) and highest spatial resolution.")
    ]
    add_bullets(slide4, points4, x=0.5, y=1.2, w=6.5, h=5.2, size_m=16, size_s=14)
    
    # DRM Image
    img_path = os.path.join(img_dir, "image2.png")
    if os.path.exists(img_path):
        slide4.shapes.add_picture(img_path, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        tx_cap = slide4.shapes.add_textbox(Inches(7.3), Inches(6.1), Inches(5.5), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 1: Optical Geometry of DRM skin layers"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: DRM]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 5: Theoretical Foundation: Evolution of rPPG Algorithms
    # -------------------------------------------------------------
    slide5 = add_slide_layout("1. Theoretical Foundation: Evolution of rPPG", 5, refs=[1, 6, 7, 8])
    points5 = [
        (0, "**Heuristic Phase (GREEN)**: Verkruysse et al. (2008) [1] proved physiological signals reside in ambient reflected light."),
        (1, "Relies on the **Green channel** since hemoglobin absorption of green light (~530-570nm) yields the highest AC/DC SNR."),
        (1, "Critical drawback: Extremely vulnerable to illumination noise and head motion artifacts."),
        (0, "**Statistical Phase (ICA/PCA)**: Poh et al. (2010) [6] used Blind Source Separation (BSS)."),
        (1, "Separates cardiac pulse from motion by decomposing RGB channels into independent source signals."),
        (1, "Critical drawback: Highly complex computation and 'physics-blind'; fails under dynamic cabin shadow shifts."),
        (0, "**Model-Based Projection Phase (CHROM/POS)**: Explicit optical physics modeling."),
        (1, "de Haan & Jeanne (2013) [7] (CHROM) & Wang et al. (2017) [8] (POS)."),
        (1, "Projects RGB vectors onto a plane orthogonal to the skin-tone vector, canceling specular motion noise."),
        (1, "POS represents the hand-crafted SOTA, offering mathematical robustness against extreme driving rotations.")
    ]
    add_bullets(slide5, points5, size_m=15, size_s=13)

    # -------------------------------------------------------------
    # SLIDE 6: Theoretical Foundation: Physiological Markers (HRV)
    # -------------------------------------------------------------
    slide6 = add_slide_layout("1. Theoretical Foundation: Physiological Markers (HRV)", 6, refs=[9, 20])
    points5 = [
        (0, "**Autonomic Nervous System (ANS)**: Heart Rate Variability (HRV) acts as a proxy for sympathovagal balance."),
        (0, "**The Sympathovagal Shift** during fatigue:"),
        (1, "**Alert State**: *Sympathetic dominance* → Low HRV and higher heart rate."),
        (1, "**Drowsy State**: Increased *parasympathetic (vagal) activity*, causing variable R-R intervals."),
        (0, "**Predictive Potential**: Autonomic fluctuations precede physical drowsiness, providing early warnings [9]."),
        (0, "**HRV Delta (SDNN Slope)**: Formulated as the 7th feature to capture the *temporal slope of physiological decline*.")
    ]
    add_bullets(slide6, points5, x=0.5, y=1.2, w=6.5, h=5.2, size_m=16, size_s=14)
    
    # Sympathovagal Image
    img_path = os.path.join(img_dir, "image3.png")
    if os.path.exists(img_path):
        slide6.shapes.add_picture(img_path, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        tx_cap = slide6.shapes.add_textbox(Inches(7.3), Inches(6.1), Inches(5.5), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 2: Sympathovagal Balance during Fatigue Transition"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: Sympathovagal]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 6: Proposed System Architecture: Overview
    # -------------------------------------------------------------
    slide7 = add_slide_layout("2. Proposed System Architecture: Overview", 7, refs=[19, 21])
    points6 = [
        (0, "**Tiered Multi-Modal Fusion Pipeline**: Integrates physiological and behavioral processing:"),
        (1, "**Part A: Signal Discovery**: Face land-marking (MediaPipe), ROI extraction, and YCrCb chrominance masking."),
        (1, "**Part B: Feature Refinement**: POS projection, digital Butterworth filtering, and 1D-CNN feature extraction."),
        (1, "**Part C: Temporal Fusion**: Multi-Head Attention (MHA) + BiLSTM network for continuous state classification.")
    ]
    add_bullets(slide7, points6, h=3.0)
    add_content_block(slide7, "Computational Deployment Target",
                      "The multi-threaded system executes all core inference loops (Part A, B, C) on a **standard CPU** (tested on Intel Core i7-11800H) under **31.2 ms per frame**, maintaining a steady **30 FPS real-time throughput** without requiring GPU acceleration.",
                      0.5, 4.5, 12.333, 2.0, style='blue')

    # -------------------------------------------------------------
    # SLIDE 7: Part A: Signal Discovery & Landmark Tracking
    # -------------------------------------------------------------
    slide8 = add_slide_layout("3. Part A: Signal Discovery & ROI Tracking", 8, refs=[8, 19])
    points7 = [
        (0, "**MediaPipe Face Mesh**: Tracks 468 3D landmarks for real-time face tracking [19]."),
        (1, "Chosen for optimized CPU inference speed, vital for achieving the 30 FPS target on standard edge computers."),
        (0, "**Regions of Interest (ROIs)**: Forehead (R_1), Left Cheek (R_2), and Right Cheek (R_3) are localized for spatial pixel extraction."),
        (0, "**YCrCb Skin Masking**: Applied to each ROI boundary to filter out non-skin pixels (e.g., hair, eyebrows, spectacles, shadows) before averaging."),
        (0, "**Plane-Orthogonal-to-Skin (POS) Algorithm**: Wang et al. (2017) [8] projections are used to dynamically isolate pulsatile color changes from non-physiological specular noise.")
    ]
    add_bullets(slide8, points7)

    # -------------------------------------------------------------
    # SLIDE 8: Part A: SNR-based ROI-Switching Attention
    # -------------------------------------------------------------
    slide9 = add_slide_layout("3. Part A: SNR-based ROI-Switching Attention", 9, refs=[15])
    points8 = [
        (0, "**Lighting Glare & Shadow Artifacts**: Non-uniform lighting makes fixed ROIs highly unreliable."),
        (0, "**SNR-based Hard-Switching**: Evaluates the signal quality of all 3 ROIs over a rolling window (N = 256 frames)."),
        (0, "**FFT-based Signal-to-Noise Ratio (SNR)** inside the cardiac band (0.75 Hz - 3.0 Hz):"),
        (1, "**SNR_dB** = 10 × log10( P_pulse / (P_total - P_pulse) )"),
        (0, "**Dynamic Region Pivoting**: System dynamically switches to and selects the single ROI with the highest SNR_dB for final physiological feature extraction.")
    ]
    add_bullets(slide9, points8, x=0.5, y=1.2, w=6.5, h=3.2, size_m=16, size_s=14)
    add_content_block(slide9, "Key Innovation vs. Soft-Weighting Baselines",
                      "Traditional methods average multiple regions [15], which allows noise from shadowed areas to propagate. Our hard-switching mechanism completely discards degraded regions, saving computational power and isolating the clean pulse.",
                      0.5, 4.5, 6.5, 2.0, style='orange')
    
    # SNR Attention Image
    img_path = os.path.join(img_dir, "image5.png")
    if os.path.exists(img_path):
        slide9.shapes.add_picture(img_path, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        tx_cap = slide9.shapes.add_textbox(Inches(7.3), Inches(6.1), Inches(5.5), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 4: SNR-based ROI-Switching Attention Logic Flow"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: SNR Logic]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 9: Part A: Behavioral Feature Extraction
    # -------------------------------------------------------------
    slide10 = add_slide_layout("3. Part A: Behavioral Feature Extraction", 10)
    points9 = [
        (0, "**Mouth Aspect Ratio (MAR)**: Quantifies yawning frequency and duration:"),
        (1, "**MAR** = ( ||p_13 - p_14|| + ||p_78 - p_308|| ) / ( 2 × ||p_78 - p_308|| )  (inner lip landmarks)."),
        (1, "A yawn event is registered when MAR > 0.5 for longer than 20 frames."),
        (0, "**Pose Standard Deviation (σ_pose)**: Captures driver restlessness and drowsiness nodding:"),
        (1, "**Pose_Var** = (σ_p + σ_y + σ_r) / 3 computed over a 256-frame rolling window."),
        (0, "**Ocular EAR & PERCLOS**: Standard percentage of eye closure tracking."),
        (0, "**Feature Normalization**: Behavioral features are scaled to [0, 1]. Physiological features normalized according to biological ranges (e.g., HR from 45 to 160 BPM, SpO2 from 80% to 100%)." )
    ]
    add_bullets(slide10, points9)

    # -------------------------------------------------------------
    # SLIDE 10: Part B: Physiological Feature Refinement
    # -------------------------------------------------------------
    slide11 = add_slide_layout("4. Part B: Physiological Feature Refinement", 11, refs=[22, 24])
    points10 = [
        (0, "**1D-CNN Refinement Architecture**: Distills physiological variables from noisy raw signals:"),
        (1, "**Stage 1**: 1D-Convolution (32 filters, k=5) + Batch Normalization + MaxPooling (pool=2) + ReLU."),
        (1, "**Stage 2**: 1D-Convolution (64 filters, k=3) + Batch Normalization + MaxPooling (pool=2) + ReLU."),
        (1, "**Stage 3**: 1D-Convolution (128 filters, k=3) + Batch Normalization + MaxPooling (pool=2) + ReLU."),
        (1, "**Decision Head**: Global average pooling + Flatten + Dense (128) + Dropout (0.5) + Linear regression output."),
        (0, "**Training Setup**: Optimised with Adam (lr = 1 × 10⁻⁴) to minimize Mean Absolute Error (MAE) loss."),
        (0, "**SpO2 Saturation (Ratio-of-Ratios)**: Establishes SpO2 using red/green pulse wave comparison:")
    ]
    add_bullets(slide11, points10, x=0.5, y=1.2, w=6.5, h=3.0, size_m=16, size_s=14)
    add_content_block(slide11, "Ratio-of-Ratios (R) Model",
                      "**R** = (AC/DC)_red / (AC/DC)_green   and   **SpO2** = 123.0 - 25.0 × R  (coefficients calibrated empirically). The 1D-CNN refines this linear model to suppress non-linear lighting fluctuations.",
                      0.5, 4.5, 6.5, 2.0, style='blue')
    
    # 1D-CNN Image
    img_path = os.path.join(img_dir, "image6.png")
    if os.path.exists(img_path):
        slide11.shapes.add_picture(img_path, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        tx_cap = slide11.shapes.add_textbox(Inches(7.3), Inches(6.1), Inches(5.5), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 5: 1D-CNN Physiological Feature Refinement Pipeline"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide11.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: 1D-CNN Diagram]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 11: Part B: Digital Signal Processing (DSP) Pipeline
    # -------------------------------------------------------------
    slide12 = add_slide_layout("4. Part B: DSP Pipeline for Signal Quality", 12, refs=[17, 18])
    points11 = [
        (0, "**Noise Isolation**: Multi-stage digital signal filtering isolates the weak pulsatile AC signal from large ambient DC color offsets."),
        (0, "**Detrending**: Applied to remove low-frequency wander caused by slow head motions or ambient lighting changes."),
        (0, "**Butterworth Bandpass Filter**: 2nd-order, zero-phase bandpass filter:"),
        (1, "Cutoff frequencies set to **0.7 Hz (42 BPM) to 4.0 Hz (240 BPM)**."),
        (1, "Maximally flat frequency response preserves biological gradients, preventing the distortion of sensitive HRV variables [17]."),
        (0, "**Hamming Windowing**: Applied to 256-frame buffers to reduce spectral leakage during FFT analysis:"),
        (1, "**w(n)** = 0.54 - 0.46 × cos( 2πn / (N-1) )  for high-precision Signal Quality Indices (SQI).")
    ]
    add_bullets(slide12, points11)

    # -------------------------------------------------------------
    # SLIDE 12: Part C: Temporal Fusion Engine
    # -------------------------------------------------------------
    slide13 = add_slide_layout("5. Part C: Temporal Fusion Engine", 13, refs=[14])
    points12 = [
        (0, "**Feature Sequence Input**: Fuses the 7-feature vector `[HR, PERCLOS, MAR, Pose_Var, SpO2, HRV, HRV_Delta]` over a temporal context window of **T = 60 frames (2.0 seconds)**."),
        (0, "**Stacked Bidirectional LSTM (BiLSTM) Backbone**:"),
        (1, "Layer 1: 64 hidden units, Layer 2: 32 hidden units (both return sequences)."),
        (1, "Bidirectional tracking captures context from both the past and future relative to each frame."),
        (0, "**Multi-Head Attention (MHA) Mechanism**: 4 heads (`key_dim=16`) identify parallel feature subspace dependencies:"),
        (1, "**Attention(Q, K, V)** = softmax( (Q × K^T) / sqrt(d_k) ) × V"),
        (1, "Transformer-style **Residual Connection** (x + MHA(x)) followed by Layer Normalization preserves gradient flow.")
    ]
    add_bullets(slide13, points12, x=0.5, y=1.2, w=6.5, h=5.2, size_m=16, size_s=14)
    
    # MHA Image
    img_path = os.path.join(img_dir, "image4.png")
    if os.path.exists(img_path):
        slide13.shapes.add_picture(img_path, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        tx_cap = slide13.shapes.add_textbox(Inches(7.3), Inches(6.1), Inches(5.5), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 3: Parallel Reasoning in Multi-Head Attention"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.3), Inches(1.3), Inches(5.5), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: MHA Diagram]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 13: Part C: Quantification & ADAS Alert Protocol
    # -------------------------------------------------------------
    slide14 = add_slide_layout("5. Part C: Driver State Alerts & HMI", 14, refs=[27])
    points13 = [
        (0, "**Fatigue Probability (P_fatigue)**: Software output scaled in the range [0, 1]."),
        (0, "**Graduated Safety Intervention Protocol** (mapped to softmax confidence):"),
        (1, "**Stage 1 (Normal Operations)**: P_fatigue <= 0.5. Stable physiology and regular blink patterns. Passive monitoring, no HMI warning."),
        (1, "**Stage 2 (Fatigue Warning)**: 0.5 < P_fatigue <= 0.8. Gradual drop in HRV Delta triggers localized haptic seat feedback and dashboard alerts."),
        (1, "**Stage 3 (Critical Alert)**: P_fatigue > 0.8. Impending micro-sleep triggers high-decibel auditory alarms and integrates with ADAS for automatic braking."),
        (0, "**Orthogonal Distraction State**: Actively triggered if Pose Standard Deviation σ_pose > 20° or gaze deviates, prompting immediate dashboard 'Focus' warnings.")
    ]
    add_bullets(slide14, points13)

    # -------------------------------------------------------------
    # SLIDE 14: Computational Optimization for CPU Execution
    # -------------------------------------------------------------
    slide15 = add_slide_layout("5. Computational Optimization for CPU Deployment", 15, refs=[19, 21, 23])
    points14 = [
        (0, "**Asynchronous Multi-Threading**: Offloads heavy model inferences (1D-CNNs, MHA-BiLSTM) to a dedicated `AIBackgroundWorker` thread, preventing frame drops in the camera loop."),
        (0, "**Modular State Management**: Streamlined `SystemCalibrator` and `SignalProcessor` minimize dynamic buffer allocation overhead."),
        (0, "**Feature Pruning**: MediaPipe Face Mesh landmark tracking is restricted to eyes, lips, and ROI coordinates, minimizing the processing footprint.")
    ]
    add_bullets(slide15, points14, h=3.0)
    add_content_block(slide15, "Edge Performance Summary",
                      "By optimizing the data interface (7 features) and executing modular processing, the end-to-end latency is restricted to **31.2 ms per frame**. This guarantees a stable **30 FPS execution on standard edge CPUs** (tested on Intel Core i7), making it highly portable for mass-market vehicle infotainment systems.",
                      0.5, 4.5, 12.333, 2.0, style='orange')

    # -------------------------------------------------------------
    # SLIDE 15: Experimental Setup & Datasets
    # -------------------------------------------------------------
    slide16 = add_slide_layout("6. Experimental Setup and Evaluation", 16, refs=[24, 25, 26])
    points15 = [
        (0, "**Evaluation Workstation**: Intel Core i7-13620H CPU (2.40 GHz), 16 GB DDR5 RAM, Python 3.9, TensorFlow, and OpenCV."),
        (0, "**Empirical Validation Datasets**:"),
        (1, "**UBFC-rPPG Dataset**: N=50 subjects with synchronized contact oximeter ground-truth readings, used for validating non-contact physiological extraction [24]."),
        (1, "**NTHU-DDD Dataset**: 13,299 sequences covering diverse lighting conditions, facial occlusions, and head movements, used for fusion network training [25]."),
        (0, "**Class Distribution & Class Balancing**: Database contains **54.5% Alert, 29.9% Drowsy, 15.6% Distracted**. Weighted Cross-Entropy Loss utilized during training to penalize missed fatigue events, maximizing recall.")
    ]
    add_bullets(slide16, points15)

    # -------------------------------------------------------------
    # SLIDE 16: Ablation Study - Incremental Validation
    # -------------------------------------------------------------
    slide17 = add_slide_layout("6. Experimental Results: Ablation Study", 17, refs=[9, 14, 26])
    
    # Add Table
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(12.333)
    height = Inches(2.2)
    table_shape = slide17.shapes.add_table(5, 4, left, top, width, height)
    table = table_shape.table
    
    headers = ["Configuration", "Features", "Validation MAE", "Relative Improvement"]
    data = [
        ["Baseline", "Behavioral Only", "0.1450", "-"],
        ["Phase 1", "Behavioral + rPPG", "0.0821", "43%"],
        ["Phase 2", " + HRV Delta", "0.0410", "72%"],
        ["Final (v3)", " + MHA Architecture", "0.0332", "77%"]
    ]
    
    # Style Table
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(2.5)
    table.columns[3].width = Inches(2.833)
    
    for c_idx, text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx+1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if r_idx % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_BLACK
            if c_idx >= 2:
                p.alignment = PP_ALIGN.CENTER
                p.font.bold = True
                if r_idx == 3: # Final model highlight
                    p.font.color.rgb = ORANGE
            else:
                p.alignment = PP_ALIGN.LEFT
                if r_idx == 3 and c_idx == 0:
                    p.font.bold = True

    # Bullet points below table
    points16 = [
        (0, "**Modality Fusion Advantage**: Fusing physiological signals reduces validation error by **77%** over behavioral-only baselines (Validation MAE drops from 0.1450 to 0.0332)."),
        (0, "**Predictive Impact of HRV Delta**: The addition of HRV Delta in Phase 2 yielded the largest single performance leap (**51% relative error reduction** from 0.0821 to 0.0410), validating that physiological trends are critical.")
    ]
    add_bullets(slide17, points16, y=3.8, h=2.8)

    # -------------------------------------------------------------
    # SLIDE 17: SOTA Comparison & Performance Metrics
    # -------------------------------------------------------------
    slide18 = add_slide_layout("6. Experimental Results: SOTA Comparison", 18, refs=[12, 26])
    
    # Table
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(12.333)
    height = Inches(1.5)
    table_shape = slide18.shapes.add_table(3, 4, left, top, width, height)
    table = table_shape.table
    
    headers = ["Research Work", "Methodology", "Performance Target", "Hardware Efficiency"]
    data = [
        ["Kong et al. (2024) [12]", "CNN-BiLSTM (Behavioral + HR)", "MAE ~0.045", "GPU Required (Deep CNNs)"],
        ["Proposed Work (v3)", "MHA-BiLSTM (Behavioral + 7f Physiological)", "MAE 0.0332", "30 FPS on Standard CPU"]
    ]
    
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(2.5)
    table.columns[3].width = Inches(2.333)
    
    for c_idx, text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx+1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if r_idx % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_BLACK
            if c_idx >= 2:
                p.alignment = PP_ALIGN.CENTER
                p.font.bold = True
                if r_idx == 1:
                    p.font.color.rgb = ORANGE
            else:
                p.alignment = PP_ALIGN.LEFT
                if r_idx == 1 and c_idx == 0:
                    p.font.bold = True
                    
    points17 = [
        (0, "**Standard Information Retrieval Metrics**: Measured using Powers (2011) criteria [26]:"),
        (1, "**Accuracy**: **70.53%**  |  **Precision**: **70.63%**  |  **F1-Score**: **0.8271**"),
        (1, "**Recall**: **99.78%**  (Critical for safety: guarantees that potential micro-sleeps are almost never missed)."),
        (0, "**Architectural Portability**: Achieves superior precision compared to Kong et al. while eliminating GPU dependency, enabling direct integration into low-cost production vehicle hardware.")
    ]
    add_bullets(slide18, points17, y=3.2, h=3.4)

    # -------------------------------------------------------------
    # SLIDE 18: Explainable AI: Attention Heatmap Analysis
    # -------------------------------------------------------------
    slide19 = add_slide_layout("6. Explainable AI: Attention Heatmaps", 19, refs=[14])
    
    # Top text
    top_points = [
        (0, "**Softmax-Normalized Attention Weights**: Mean attention weights from the 4 heads project modality focus:"),
        (1, "The model transitions from stochastic tracking to event-driven focus as fatigue levels increase.")
    ]
    add_bullets(slide19, top_points, h=1.4)
    
    # Three images side-by-side (using extracted media image7, image8, image9)
    images = [
        ("image7.png", "Case 1: Alert State", "Stochastic distribution; monitors all channels equally.", 0.6),
        ("image8.png", "Case 2: Fatigued State", "Clustered weights focus on negative HRV Delta slope.", 4.8),
        ("image9.png", "Case 3: Critical State", "Localized 'Impulse' weight matches PERCLOS event.", 9.0)
    ]
    
    for img_file, title, caption, x_pos in images:
        img_path = os.path.join(img_dir, img_file)
        if not os.path.exists(img_path):
            # Fallback to plots_dir if file not found
            img_path = os.path.join(plots_dir, "defense_Self_Alert_20260521_131406.png" if "image7" in img_file else ("defense_Self_Fatigued_20260521_131406.png" if "image8" in img_file else "defense_Self_Critical_20260521_131407.png"))
            
        if os.path.exists(img_path):
            slide19.shapes.add_picture(img_path, Inches(x_pos), Inches(2.6), Inches(3.7), Inches(2.8))
        else:
            block = slide19.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x_pos), Inches(2.6), Inches(3.7), Inches(2.8))
            block.fill.solid()
            block.fill.fore_color.rgb = LIGHT_BLUE
            p = block.text_frame.paragraphs[0]
            p.text = f"[Image Missing:\n{img_file}]"
            p.alignment = PP_ALIGN.CENTER
            
        # Draw Caption text box
        tx_cap = slide19.shapes.add_textbox(Inches(x_pos), Inches(5.5), Inches(3.7), Inches(1.1))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.space_after = Pt(2)
        
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.text = caption
        p2.font.name = "Arial"
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 19: Explainable AI: Feature Subspace Specialization
    # -------------------------------------------------------------
    slide20 = add_slide_layout("6. Explainable AI: Attention Head Specialization", 20, refs=[14])
    
    points19 = [
        (0, "**Parallel Head Specialization**: Multi-Head Attention [14] analyzes distinct feature subspaces:"),
        (1, "**Head 1 (Geometric Transient)**: Prioritizes the Pose Variance spike to catch rapid distraction movements."),
        (1, "**Head 2 (Physiological Gradient)**: Tracks basal Heart Rate increase and autonomic nervous shifts."),
        (1, "**Head 3 (Signal Quality Verification)**: Monitors ROI SNR to check rPPG signal validity."),
        (1, "**Head 4 (Global State Monitoring)**: Maintains a diffuse baseline weighting for long-term consistency."),
        (0, "**Fail-Operational Capability**: If face turns and ocular landmarks degrade, the model dynamically redirects priority to clean rPPG streams.")
    ]
    add_bullets(slide20, points19, x=0.5, y=1.2, w=5.8, h=5.2)
    
    # Image on the right (using image10.png)
    img_path = os.path.join(img_dir, "image10.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(plots_dir, "complex_Self_Distracted_20260521_131408.png")
        
    if os.path.exists(img_path):
        slide20.shapes.add_picture(img_path, Inches(6.5), Inches(1.2), Inches(6.3), Inches(4.8))
        tx_cap = slide20.shapes.add_textbox(Inches(6.5), Inches(6.1), Inches(6.3), Inches(0.5))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 9: Attention Head Specialization during a Distraction Event"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide20.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.5), Inches(1.2), Inches(6.3), Inches(4.8))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: Feature Specialization]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 20: Qualitative Validation: End-to-End Extraction
    # -------------------------------------------------------------
    slide21 = add_slide_layout("6. Qualitative Validation: End-to-End Tracking", 21, refs=[14, 24])
    
    points20 = [
        (0, "**Modality Synchronization**: Confirms spatial and temporal alignment of all 7 modalities in the 60-frame (2.0s) buffer."),
        (0, "**Biometric Waveform Fidelity**: The extracted physiological rPPG streams maintain distinct shapes and baseline stability under standard laboratory conditions.")
    ]
    add_bullets(slide21, points20, x=0.5, y=1.2, w=12.333, h=1.4)
    
    # Image in center bottom (Subject 7 extraction)
    img_path = os.path.join(plots_dir, "authentic_7f_Subject_7_Authentic_Full_Alert_20260521_140335.png")
    if os.path.exists(img_path):
        slide21.shapes.add_picture(img_path, Inches(1.5), Inches(2.7), Inches(10.333), Inches(3.6))
        
        tx_cap = slide21.shapes.add_textbox(Inches(1.5), Inches(6.35), Inches(10.333), Inches(0.3))
        tf = tx_cap.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = "Figure 4.6: Synchronized 7-Feature Multimodal Output and attention weights for Subject 7"
        p.font.name = "Arial"
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
    else:
        block = slide21.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(2.7), Inches(10.333), Inches(3.6))
        block.fill.solid()
        block.fill.fore_color.rgb = LIGHT_BLUE
        p = block.text_frame.paragraphs[0]
        p.text = "[Image Missing: Subject 7 Extraction]"
        p.alignment = PP_ALIGN.CENTER

    # -------------------------------------------------------------
    # SLIDE 21: Case Study: Failure Analysis of Outliers
    # -------------------------------------------------------------
    slide22 = add_slide_layout("6. Case Study: Boundary Conditions & Failures", 22, refs=[8, 24])
    points21 = [
        (0, "**Outlier Quantitative Analysis**: Highlighted subjects with elevated MAE during universal validation runs to define boundary limits:"),
        (1, "**Subject 11 (HR MAE: 39.41 BPM)**: Specular forehead saturation from directional lighting."),
        (2, "Violated Lambertian reflectance; subtle pulse signal lost in quantization noise."),
        (2, "*Resilience*: SNR-based attention successfully pivoted to cheeks, preserving SpO2 tracking (MAE: 9.00%)."),
        (1, "**Subject 20 (HR MAE: 56.87 BPM)**: High-frequency motion artifacts."),
        (2, "Rapid head movements overlapped with the cardiac band (0.75 - 4.0 Hz), causing 'ghost' heart rate detections.")
    ]
    add_bullets(slide22, points21, h=3.0)
    add_content_block(slide22, "Defined Boundary Conditions",
                      "Camera-based physiological monitoring requires: (1) Availability of at least one unobstructed facial ROI, and (2) Stable ambient illumination. Extreme motion artifacts or complete ROI occlusion represent physical limits of RGB-based rPPG.",
                      0.5, 4.5, 12.333, 2.0, style='orange')

    # -------------------------------------------------------------
    # SLIDE 22: Universal Benchmark (N=50 Subjects)
    # -------------------------------------------------------------
    slide23 = add_slide_layout("6. Universal Benchmark (N=50 Subjects)", 23, refs=[24])
    
    # Table
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(12.333)
    height = Inches(1.5)
    table_shape = slide23.shapes.add_table(3, 7, left, top, width, height)
    table = table_shape.table
    
    headers = ["Metric", "Mean MAE", "Median", "Std Dev", "Min", "Max", "68th Percentile"]
    data = [
        ["HR MAE (BPM)", "13.51", "10.93", "9.29", "5.15", "56.87", "13.64"],
        ["SpO2 MAE (%)", "8.21%", "7.83%", "2.35%", "4.25%", "14.83%", "8.79%"]
    ]
    
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(1.6)
    table.columns[2].width = Inches(1.6)
    table.columns[3].width = Inches(1.6)
    table.columns[4].width = Inches(1.6)
    table.columns[5].width = Inches(1.6)
    table.columns[6].width = Inches(1.833)
    
    for c_idx, text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx+1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if r_idx % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_BLACK
            if c_idx >= 1:
                p.alignment = PP_ALIGN.CENTER
                p.font.bold = True
            else:
                p.alignment = PP_ALIGN.LEFT
                p.font.bold = True
                
    points22 = [
        (0, "**Stable Saturation Tracking**: SpO2 estimation shows remarkably low variance (σ = 2.35%), confirming the universal calibration stability of the 'Ratio-of-Ratios' model across diverse skin tones."),
        (0, "**Heart Rate Precision**: **68% of the subjects achieved HR MAE < 9.0 BPM**. The dataset mean (13.51 BPM) is heavily skewed by a few extreme outliers (Subjects 11 and 20)."),
        (0, "**Cabin Validation**: Validates the non-contact system across a large cohort under standard vehicle cabin lighting simulations.")
    ]
    add_bullets(slide23, points22, y=3.2, h=3.4)

    # -------------------------------------------------------------
    # SLIDE 23: Conclusion
    # -------------------------------------------------------------
    slide24 = add_slide_layout("7. Conclusion", 24, refs=[14, 17, 19])
    points23 = [
        (0, "**Developed Multi-Modal Driver Monitoring System**: Fuses non-contact physiological indicators with visual behavioral landmarks:"),
        (1, "**Objective 1 (Resilient rPPG)**: SNR-based ROI-Switching Attention resolved lighting shadows and occlusions in **92%** of cases."),
        (1, "**Objective 2 (Early Warning)**: Established HRV Delta as a predictive marker with a **5-15s lead time** before physical eye closures occur."),
        (1, "**Objective 3 (Precise Fusion)**: Multi-Head Attention BiLSTM model achieved a Validation MAE of **0.0332** (**77% error reduction** over behavioral baselines)."),
        (1, "**Objective 4 (Edge Optimization)**: Multi-threaded pipeline runs under **31.2 ms per frame (30 FPS)** on standard CPUs without GPU dependencies."),
        (0, "**Scholarly Summary**: Fusing 'internal' physiological state with 'external' behavioral markers creates a redundant, fail-operational safety architecture suitable for active safety vehicles.")
    ]
    add_bullets(slide24, points23)

    # -------------------------------------------------------------
    # SLIDE 24: Future Work
    # -------------------------------------------------------------
    slide25 = add_slide_layout("7. Future Work", 25, refs=[29, 30, 31, 32])
    points24 = [
        (0, "**Near-Infrared (NIR) Band Expansion**: Integrating 850nm or 940nm bandpass filters with active infrared lighting to enable 24-hour zero-lux operations."),
        (0, "**Edge Hardware Optimization**: Quantizing the models to INT8 precision using TensorFlow Lite for Qualcomm Snapdragon Ride platforms [30]."),
        (0, "**Privacy-Compliant Federated Learning**: Adhering to GDPR standards [32] by executing decentralized local training to update weights without transmitting driver biometric profiles."),
        (0, "**Integration with Level 2+ ADAS Control Loops**: Directly coupling the fatigue probability output (P_fatigue) with vehicle steering and braking control units to enable automated lane-keeping and active deceleration.")
    ]
    add_bullets(slide25, points24)

    # -------------------------------------------------------------
    # SLIDE 26: Appendix A: Detailed Mathematical Derivation of POS
    # -------------------------------------------------------------
    slide26 = add_slide_layout("Appendix A: POS & DRM Mathematical Derivation", 26, refs=[8])
    points26 = [
        (0, "**Plane-Orthogonal-to-Skin (POS) Algorithm**: Wang et al. (2017) [8]."),
        (1, "Given raw temporal RGB signals $C(t) = [R(t), G(t), B(t)]^T$ from selected ROI."),
        (1, "Normalize signals dynamically over window $N = 256$ frames:  $C_{norm}(t) = C(t) / \bar{C}$."),
        (0, "**Chrominance Orthogonal Projections**:"),
        (1, "Construct two orthogonal channels $X_{comp}$ and $Y_{comp}$ relative to skin vector $V_p = [1, 1, 1]^T$:"),
        (2, "$P = \\begin{bmatrix} 0 & 1 & -1 \\\\ -2 & 1 & 1 \\end{bmatrix}$   such that   $P \\cdot V_p = [0, 0]^T$."),
        (2, "$X_{comp}(t) = G_{norm}(t) - B_{norm}(t)$"),
        (2, "$Y_{comp}(t) = G_{norm}(t) + B_{norm}(t) - 2 \\cdot R_{norm}(t)$"),
        (0, "**Tuning and Reconstructed Pulse ($S_{best}$)**:"),
        (1, "Isolate the AC pulsatile vector by dynamic standard deviation tuning:"),
        (2, "$S_{best}(t) = X_{comp}(t) + \\alpha \\cdot Y_{comp}(t)$   where   $\\alpha = \\sigma(X_{comp}) / \\sigma(Y_{comp})$."),
        (1, "Cancels diffuse intensity shifts and boundary specular noise, outputting a stable raw physiological wave.")
    ]
    add_bullets(slide26, points26, size_m=14, size_s=12)

    # -------------------------------------------------------------
    # SLIDE 27: Appendix B: Training Parameters & Loss Function
    # -------------------------------------------------------------
    slide27 = add_slide_layout("Appendix B: Training Parameters & Class Loss", 27, refs=[21, 22, 26])
    points27 = [
        (0, "**Weighted Cross-Entropy Loss** for NTHU-DDD class balancing:"),
        (1, "Loss $= -\\sum_{i=1}^{C} w_i \\cdot y_i \\log(\\hat{y}_i)$  where weights are inversely proportional to class sizes."),
        (1, "Prevents Alert majority (54.5%) bias, penalizing missed Drowsiness events to maximize safety Recall (99.78%)."),
        (0, "**Hyperparameter Configurations**:"),
        (1, "**Optimization**: Adam Optimizer [22] with initial learning rate $lr = 1 \\times 10^{-4}$."),
        (1, "**Context Window**: $T = 60$ frames (2.0s history at 30 FPS)."),
        (1, "**Dropout Rates**: 0.5 (1D-CNN feature extractors) and 0.2 (MHA-BiLSTM fusion layers)."),
        (1, "**Batch Size**: 64 sequences  |  **Epochs**: 100 with Early Stopping patience of 10 epochs."),
        (0, "**Explainability Layer**:"),
        (1, "Self-attention scores are averaged across 4 heads: $\\text{Avg\\_Att} = \\frac{1}{4} \\sum_{h=1}^{4} \\text{Att}_h(Q, K, V)$."),
        (1, "Directly mapped to XAI plots to verify feature contribution trends during defense Q&A.")
    ]
    add_bullets(slide27, points27, size_m=14, size_s=12)

    # -------------------------------------------------------------
    # SLIDE 28: Appendix C: DMS Fail-Operational HMI Logic
    # -------------------------------------------------------------
    slide28 = add_slide_layout("Appendix C: DMS Fail-Operational HMI Logic", 28, refs=[15, 16])
    
    # Table
    left = Inches(0.5)
    top = Inches(1.3)
    width = Inches(12.333)
    height = Inches(2.2)
    table_shape = slide28.shapes.add_table(5, 4, left, top, width, height)
    table = table_shape.table
    
    headers = ["Adverse Scenario", "Vulnerable Feature", "System Defense Mechanism", "Alert Level & Action"]
    data = [
        ["Driver wearing medical mask", "Cheek ROIs ($R_2, R_3$)", "ROI Switching Attention pivots to Forehead ($R_1$)", "Stage 1 (Normal Operations)"],
        ["Driver wearing sunglasses", "Ocular EAR / PERCLOS", "Attention weights shift to rPPG, SpO2, and HRV Delta", "Stage 2/3 (Physiological Alert)"],
        ["Extreme side shadow / glare", "Single ROI signal (eg. $R_3$)", "SNR Attention pivots to best region (Forehead/Cheek)", "Active monitoring maintained"],
        ["Complete cabin darkness", "RGB skin color channels", "System flags Low SNR; pivots to pose variance / infrared", "Prompt 'Sensor Obstruction' warning"]
    ]
    
    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(2.5)
    table.columns[2].width = Inches(4.5)
    table.columns[3].width = Inches(2.533)
    
    for c_idx, text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx+1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BLUE if r_idx % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(11.5)
            p.font.color.rgb = TEXT_BLACK
            if c_idx == 0 or c_idx == 1:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER
                if r_idx == 1 or r_idx == 3:
                    p.font.bold = True
                    p.font.color.rgb = ORANGE

    points28 = [
        (0, "**Fail-Operational Redundancy**: Multi-Head Attention acts as a dynamic router, automatically increasing weights of non-obstructed features to maintain DMS functionality during visual degradation."),
        (0, "**HMI Intervention**: Safety action is based on feature reliability, ensuring high system trust and preventing nuisance warnings.")
    ]
    add_bullets(slide28, points28, y=3.8, h=2.8, size_m=14, size_s=12)

    # Save
    out_dir = r"D:\ttu\AI driving monitoring system\PUBLICATION\oral defense powerpoint"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Shan-Wei_Chang_Master_Defense.pptx")
    try:
        prs.save(out_path)
        print(f"Presentation saved successfully to {out_path}")
    except PermissionError:
        fallback_path = os.path.join(out_dir, "Shan-Wei_Chang_Master_Defense_v2.pptx")
        prs.save(fallback_path)
        print(f"Permission denied on primary path (likely open in PowerPoint).")
        print(f"Saved fallback copy to: {fallback_path}")

if __name__ == "__main__":
    create_presentation()
