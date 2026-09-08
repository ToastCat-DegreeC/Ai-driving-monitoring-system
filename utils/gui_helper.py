import cv2
import numpy as np
import config

def draw_dashboard(frame, data):
    """
    Draws a professional side-panel dashboard.
    """
    h, w, _ = frame.shape
    dash_w = config.DASHBOARD_WIDTH
    
    # Create background for dashboard
    dash_panel = np.full((h, dash_w, 3), config.COLOR_BG, dtype=np.uint8)
    
    # --- 1. Header ---
    cv2.putText(dash_panel, "DRIVER MONITOR", (20, 35), 0, 0.7, config.COLOR_TEXT, 2)
    cv2.line(dash_panel, (20, 48), (dash_w-20, 48), (100, 100, 100), 1)

    # --- 2. Vitals Section ---
    y_ptr = 75
    draw_data_card(dash_panel, "HEART RATE", f"{data['hr']:.0f}" if data['valid'] else "---", "BPM", 20, y_ptr)
    
    y_ptr += 68
    draw_data_card(dash_panel, "HRV (SDNN)", f"{data['hrv']:.1f}" if data['valid'] else "---", "ms", 20, y_ptr)
    
    y_ptr += 68
    draw_data_card(dash_panel, "OXYGEN (SpO2)", f"{data['spo2']:.1f}%" if data['valid'] else "---", "", 20, y_ptr)

    # --- 3. Pulse Wave Graph ---
    y_ptr += 75
    draw_graph(dash_panel, data['pulse_wave'], 20, y_ptr, dash_w-40, 55)

    # --- 4. Fatigue Meter ---
    y_ptr += 85
    draw_fatigue_meter(dash_panel, data['fatigue_prob'], 20, y_ptr, dash_w-40)

    # --- 5. Status Indicators ---
    y_ptr += 45
    st_col = config.COLOR_ACCENT if "Focused" in data['distraction'] else config.COLOR_ALERT
    cv2.putText(dash_panel, "ATTENTION:", (20, y_ptr), 0, 0.4, (150, 150, 150), 1)
    cv2.putText(dash_panel, data['distraction'].upper(), (20, y_ptr + 18), 0, 0.55, st_col, 2)
    
    y_ptr += 42
    eye_closed = data.get('eye_closed', False)
    eye_col = config.COLOR_ALERT if eye_closed else config.COLOR_ACCENT
    eye_str = "CLOSED" if eye_closed else "OPEN"
    cv2.putText(dash_panel, "OCULAR STATE:", (20, y_ptr), 0, 0.4, (150, 150, 150), 1)
    cv2.putText(dash_panel, f"{eye_str} (PERCLOS: {int(data.get('perclos', 0.0)*100)}%)", (20, y_ptr + 18), 0, 0.55, eye_col, 2)

    y_ptr += 42
    alert_col = config.COLOR_ALERT if "CRITICAL" in data['status'].upper() or "FATIGUED" in data['status'].upper() else config.COLOR_TEXT
    cv2.putText(dash_panel, "ALERTNESS:", (20, y_ptr), 0, 0.4, (150, 150, 150), 1)
    cv2.putText(dash_panel, data['status'].upper(), (20, y_ptr + 18), 0, 0.55, alert_col, 2)

    # --- 6. Attention Weights Visualization (Interpretability) ---
    y_ptr += 48
    att_weights = data.get('attention_weights', [])
    draw_attention_weights(dash_panel, att_weights, 20, y_ptr, dash_w-40, 45)

    # Combine frame and dashboard
    combined = np.hstack((frame, dash_panel))
    return combined

def draw_data_card(panel, label, value, unit, x, y):
    cv2.putText(panel, label, (x, y), 0, 0.4, (150, 150, 150), 1)
    cv2.putText(panel, value, (x, y + 28), 0, 0.9, config.COLOR_ACCENT, 2)
    if unit:
        cv2.putText(panel, unit, (x + 80, y + 28), 0, 0.4, (150, 150, 150), 1)

def draw_graph(panel, signal, x, y, w, h):
    cv2.rectangle(panel, (x, y), (x+w, y+h), (40, 40, 40), -1)
    cv2.putText(panel, "PULSE WAVE", (x, y - 8), 0, 0.38, (150, 150, 150), 1)
    if len(signal) < 2: return
    sig = np.array(signal)
    s_min, s_max = np.min(sig), np.max(sig)
    if s_max - s_min > 1e-6: sig = (sig - s_min) / (s_max - s_min)
    pts = []
    for i in range(len(sig)):
        pts.append([x + int(i * (w / len(sig))), y + h - int(sig[i] * h)])
    cv2.polylines(panel, [np.array(pts)], False, config.COLOR_GRAPH, 1)

def draw_fatigue_meter(panel, prob, x, y, w):
    cv2.putText(panel, "FATIGUE PROBABILITY", (x, y - 8), 0, 0.38, (150, 150, 150), 1)
    cv2.rectangle(panel, (x, y), (x+w, y+16), (40, 40, 40), -1)
    fill_w = int(np.clip(prob, 0, 1) * w)
    color = config.COLOR_ACCENT # Green
    if prob > config.FATIGUE_THRESHOLD_FATIGUED: color = (0, 255, 255) # Yellow
    if prob > config.FATIGUE_THRESHOLD_CRITICAL: color = config.COLOR_ALERT # Red
    cv2.rectangle(panel, (x, y), (x+fill_w, y+16), color, -1)
    cv2.putText(panel, f"{int(prob*100)}%", (x + w - 38, y + 13), 0, 0.4, config.COLOR_TEXT, 1)

def draw_attention_weights(panel, weights, x, y, w, h):
    """
    Draws a small bar chart of attention weights (interpretability) with timeline annotations.
    """
    # Header with XAI badge and Y-axis indicator
    cv2.putText(panel, "TEMPORAL ATTENTION", (x, y - 8), 0, 0.38, (150, 150, 150), 1)
    cv2.putText(panel, "[MHA XAI]", (x + 132, y - 8), 0, 0.32, (0, 200, 255), 1)
    cv2.putText(panel, "Weight", (x + w - 42, y - 8), 0, 0.30, (120, 120, 120), 1)

    cv2.rectangle(panel, (x, y), (x+w, y+h), (40, 40, 40), -1)
    
    # Subtle 50% guide line
    cv2.line(panel, (x, y + int(h/2)), (x+w, y + int(h/2)), (50, 50, 50), 1)
    
    weights = np.array(weights) if weights is not None else np.array([])
    if len(weights) == 0 or np.all(weights == 0):
        cv2.putText(panel, "AWAITING ATTENTION FOCUS...", (x + 25, y + int(h/2) + 4), 0, 0.35, (110, 110, 110), 1)
    else:
        max_w = np.max(weights) if np.max(weights) > 0 else 1.0
        bar_w = w / len(weights)
        for i in range(len(weights)):
            bar_h = int((weights[i] / max_w) * h)
            # Highlight highest focus frames in coral/amber, routine in cyan
            color = (0, 165, 255) if weights[i] > 0.65 * max_w else (0, 200, 255)
            cv2.rectangle(panel, 
                          (int(x + i*bar_w), y + h - bar_h), 
                          (int(x + (i+1)*bar_w) - 1, y + h), 
                          color, -1)

    # Time Axis Annotations below the box
    cv2.putText(panel, "-2.0s (Past)", (x, y + h + 15), 0, 0.32, (130, 130, 130), 1)
    cv2.putText(panel, "Time (60 Frames)", (x + int(w/2) - 45, y + h + 15), 0, 0.30, (90, 90, 90), 1)
    cv2.putText(panel, "Now", (x + w - 28, y + h + 15), 0, 0.32, (130, 130, 130), 1)
