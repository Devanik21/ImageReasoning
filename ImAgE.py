import streamlit as st
from streamlit_drawable_canvas import st_canvas
import google.generativeai as genai
from PIL import Image
import json
import io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import base64

# Page config
st.set_page_config(
    page_title="Student-Teacher Visual Feature Learning",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'iteration_history' not in st.session_state:
    st.session_state.iteration_history = []
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'student_model' not in st.session_state:
    st.session_state.student_model = None
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'canvas_image' not in st.session_state:
    st.session_state.canvas_image = None

# ================================
# STUDENT MODEL: CNN-based Feature Extractor
# ================================
class StudentCNN(nn.Module):
    """Student model: CNN that extracts visual features from images"""
    def __init__(self, num_features=10):
        super(StudentCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Feature extraction heads
        # Each head predicts one visual feature type
        self.feature_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_features)
        ])
        
        # Feature description generator (simplified)
        self.description_embeddings = nn.Embedding(num_features, 256)
        
    def forward(self, x):
        # Convolutional feature extraction
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Extract features from each head
        features = []
        for head in self.feature_heads:
            feature_value = head(x)
            features.append(feature_value)
        
        return torch.cat(features, dim=1)
    
    def generate_feature_descriptions(self, image_tensor):
        """Generate textual descriptions of detected features"""
        with torch.no_grad():
            features = self.forward(image_tensor)
            
        feature_names = [
            "Main Object Presence",
            "Dominant Color",
            "Background Type",
            "Lighting Condition",
            "Composition Balance",
            "Texture Quality",
            "Spatial Layout",
            "Style Category",
            "Color Saturation",
            "Edge Complexity"
        ]
        
        descriptions = []
        for i, (name, value) in enumerate(zip(feature_names, features[0])):
            confidence = torch.sigmoid(value).item()
            descriptions.append({
                'index': i + 1,
                'name': name,
                'confidence': confidence,
                'description': self._interpret_feature(name, confidence)
            })
        
        return descriptions
    
    def _interpret_feature(self, feature_name, confidence):
        """Interpret feature value into human-readable description"""
        if "Color" in feature_name:
            if confidence > 0.7:
                return f"Strong presence of {feature_name.lower()} detected with high confidence"
            elif confidence > 0.4:
                return f"Moderate {feature_name.lower()} detected"
            else:
                return f"Low {feature_name.lower()} presence"
        elif "Object" in feature_name:
            if confidence > 0.6:
                return "Clear main object identified in the image"
            else:
                return "Main object is less distinct or multiple objects present"
        elif "Lighting" in feature_name:
            if confidence > 0.6:
                return "Bright lighting conditions"
            else:
                return "Dim or dramatic lighting"
        else:
            return f"{feature_name} score: {confidence:.3f}"

def initialize_student_model(num_features=10):
    """Initialize or reset the student model"""
    model = StudentCNN(num_features=num_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

# ================================
# TEACHER PROMPT TEMPLATE
# ================================
TEACHER_PROMPT_TEMPLATE = """You are an expert TEACHER model (Gemma 3 27B) supervising a STUDENT CNN model in a self-training loop.
Your job is to:
1. Look at the input image.
2. Infer the 10 most important basic visual features of that image.
3. Compare those features and reasoning steps with the STUDENT's attempt.
4. Provide strict, step-by-step feedback and a numeric reward.

--------------------
ROLE & GOAL
--------------------
- You are the TEACHER (expert vision model).
- The STUDENT is a CNN model that:
  - Sees the same image.
  - Predicts 10 basic visual features with confidence scores.
  - Generates descriptions for each feature.
- Your goal is to:
  - Check every STUDENT prediction.
  - Identify any incorrect or poorly calibrated prediction.
  - Assign a global reward: +1, 0.5, or -1.

--------------------
TASK DEFINITION
--------------------
Current task name: {TASK_NAME}

For this task, a "basic visual feature" can include:
- Main objects present
- Dominant colors
- Overall layout or composition
- Background characteristics
- Lighting / shadows
- Texture or style (e.g., cartoon, photo, sketch)
- Presence of text or symbols
- Any other simple, stable property directly visible

You must infer **exactly 10** such basic visual features that are:
- High-level, easy to learn
- Directly grounded in the image
- Non-overlapping as much as possible

--------------------
INPUTS
--------------------
1. IMAGE INPUT:
   - You receive the raw image via the API (multimodal input).

2. STUDENT OUTPUT:
   The Student's CNN predictions:

   <STUDENT_OUTPUT_BEGIN>
   {STUDENT_OUTPUT}
   <STUDENT_OUTPUT_END>

The student output includes:
- Feature name
- Confidence score (0-1)
- Generated description

You MUST treat this as a fallible attempt that can contain:
- Incorrect feature detection
- Poor confidence calibration
- Hallucinated or vague descriptions

--------------------
WHAT YOU MUST DO
--------------------

Phase 1 — Teacher's own understanding:
- Carefully inspect the image.
- Infer the 10 best "ground-truth" basic visual features.

Phase 2 — Align Student features against Teacher features:
- For each student prediction:
  - Decide if it is:
    - EXACT_MATCH: Correct and well-calibrated
    - PARTIAL: Mostly correct but confidence off or minor issues
    - WRONG: Incorrect detection or major misalignment

Phase 3 — Global reward:
Assign reward as follows:

- Reward = +1
  - All predictions are correct or at most very minor issues
  - Confidence scores are well-calibrated
  - No hallucinated features

- Reward = 0.5
  - More than half predictions are correct
  - Some calibration issues but overall usable
  - At most 1-2 wrong predictions

- Reward = -1
  - Majority of predictions are wrong
  - Critical hallucinations about objects or colors
  - Poor confidence calibration

You MUST choose EXACTLY one of: 1, 0.5, -1.

--------------------
OUTPUT FORMAT (VERY IMPORTANT)
--------------------
You MUST output ONLY a single JSON object:

{{
  "teacher_features": [
    "feature_1 (your ground-truth understanding)",
    "feature_2",
    ...
    "feature_10"
  ],
  "step_feedback": [
    {{
      "index": 1,
      "student_step": "Feature name: description (confidence: X)",
      "label": "EXACT_MATCH | PARTIAL | WRONG",
      "comment": "Short explanation in 1-3 sentences."
    }}
  ],
  "overall_reward": -1,
  "reward_explanation": "1-3 sentences explaining why this reward was chosen.",
  "advice_to_student": "2-5 sentences giving actionable guidance to improve next time."
}}

Rules:
- "teacher_features" MUST have exactly 10 items
- "overall_reward" MUST be one of: -1, 0.5, 1 (as a number)
- Do NOT include any extra keys
- Do NOT output any text before or after the JSON
"""

# ================================
# IMAGE PREPROCESSING
# ================================
def preprocess_image(image, size=(224, 224)):
    """Preprocess image for student CNN model"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ================================
# STUDENT MODEL INFERENCE
# ================================
def call_student_model(image, model):
    """Run student CNN model on image"""
    try:
        # Preprocess
        image_tensor = preprocess_image(image)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            descriptions = model.generate_feature_descriptions(image_tensor)
        
        # Format output
        output_text = "STUDENT CNN PREDICTIONS:\n\n"
        for desc in descriptions:
            output_text += f"{desc['index']}. {desc['name']}\n"
            output_text += f"   Confidence: {desc['confidence']:.3f}\n"
            output_text += f"   Description: {desc['description']}\n\n"
        
        return output_text, descriptions
    except Exception as e:
        return f"Error in student model: {str(e)}", []

# ================================
# TEACHER MODEL (GEMMA)
# ================================
def call_teacher_model(image, student_output, task_name, api_key):
    """Call Gemma 3 27B as teacher/verifier"""
    try:
        genai.configure(api_key=api_key)
        
        # Use Gemma 3 27B
        model = genai.GenerativeModel('gemma-3-27b-it')
        
        prompt = TEACHER_PROMPT_TEMPLATE.format(
            TASK_NAME=task_name,
            STUDENT_OUTPUT=student_output
        )
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
        
        return json.loads(response_text)
    except Exception as e:
        return {"error": str(e)}

# ================================
# TRAINING STEP
# ================================
def train_student_step(model, optimizer, image, reward):
    """Update student model based on teacher reward"""
    try:
        image_tensor = preprocess_image(image)
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        features = model(image_tensor)
        
        # Simple reward-based loss (higher reward = lower loss)
        # In real RL, this would be PPO/REINFORCE
        loss = -reward * torch.mean(features)  # Maximize reward
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    except Exception as e:
        return 0.0

# ================================
# STREAMLIT UI
# ================================

# Sidebar controls
st.sidebar.title("⚙️ Configuration")

# API Key
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")

st.sidebar.markdown("---")

# Parameters
st.sidebar.subheader("Training Parameters")
num_iterations = st.sidebar.slider("Number of Iterations", 1, 50, 10)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
num_features = st.sidebar.number_input("Number of Features", 5, 20, 10)
task_name = st.sidebar.text_input("Task Name", "Visual Feature Extraction")

st.sidebar.markdown("---")

# Display settings
st.sidebar.subheader("Display Settings")
show_teacher_features = st.sidebar.checkbox("Show Teacher's Ground Truth", True)
show_step_feedback = st.sidebar.checkbox("Show Step-by-Step Feedback", True)
show_raw_json = st.sidebar.checkbox("Show Raw JSON Response", False)
show_loss_curve = st.sidebar.checkbox("Show Loss Curve", True)

st.sidebar.markdown("---")

# Model controls
st.sidebar.subheader("Model Controls")
if st.sidebar.button("🔄 Initialize/Reset Student Model", use_container_width=True):
    st.session_state.student_model, st.session_state.optimizer = initialize_student_model(num_features)
    st.sidebar.success("Student model initialized!")

st.sidebar.markdown("---")

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    run_button = st.button("▶️ Run Training", type="primary", use_container_width=True)
with col2:
    reset_button = st.button("🗑️ Clear History", use_container_width=True)

if reset_button:
    st.session_state.iteration_history = []
    st.session_state.current_iteration = 0
    st.rerun()

# Main title
st.title("🎓 Student-Teacher Visual Feature Learning System")
st.markdown("*CNN Student + Gemma 3 27B Teacher | Self-Verifiable Learning*")

# Canvas for drawing
tab1, tab2 = st.tabs(["🎨 Draw/Upload Image", "📊 Training Dashboard"])

with tab1:
    st.subheader("Create or Upload Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        drawing_mode = st.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon")
        )
        
        stroke_width = st.slider("Stroke width:", 1, 25, 3)
        stroke_color = st.color_picker("Stroke color:", "#000000")
        bg_color = st.color_picker("Background color:", "#FFFFFF")
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=400,
            width=600,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        
        # Save canvas image
        if canvas_result.image_data is not None:
            st.session_state.canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    with col2:
        st.markdown("**Or upload an image:**")
        uploaded_file = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'webp'])
        
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert('RGB')
            st.image(uploaded_image, caption="Uploaded", use_container_width=True)
            st.session_state.canvas_image = uploaded_image
        
        if st.session_state.canvas_image:
            st.markdown("**Current Image:**")
            st.image(st.session_state.canvas_image, use_container_width=True)

# Get active image
active_image = st.session_state.canvas_image
if active_image and active_image.mode == 'RGBA':
    # Convert RGBA to RGB
    rgb_image = Image.new('RGB', active_image.size, (255, 255, 255))
    rgb_image.paste(active_image, mask=active_image.split()[3])
    active_image = rgb_image

# Run training
if run_button and active_image and api_key:
    if st.session_state.student_model is None:
        st.session_state.student_model, st.session_state.optimizer = initialize_student_model(num_features)
    
    with tab2:
        st.header("🔄 Training Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        losses = []
        rewards = []
        
        for iteration in range(num_iterations):
            status_text.markdown(f"**Iteration {iteration + 1} / {num_iterations}**")
            
            with st.expander(f"📊 Iteration {iteration + 1}", expanded=(iteration == num_iterations - 1)):
                col1, col2 = st.columns(2)
                
                # Student phase
                with col1:
                    st.subheader("👨‍🎓 Student CNN Output")
                    with st.spinner("Student analyzing..."):
                        student_output, student_descriptions = call_student_model(
                            active_image, 
                            st.session_state.student_model
                        )
                        time.sleep(0.3)
                    
                    st.text_area(
                        "Predictions",
                        student_output,
                        height=350,
                        key=f"student_{iteration}",
                        label_visibility="collapsed"
                    )
                
                # Teacher phase
                with col2:
                    st.subheader("👨‍🏫 Teacher Evaluation (Gemma 3)")
                    with st.spinner("Teacher evaluating..."):
                        teacher_response = call_teacher_model(
                            active_image, 
                            student_output, 
                            task_name, 
                            api_key
                        )
                        time.sleep(0.3)
                    
                    if "error" in teacher_response:
                        st.error(f"Error: {teacher_response['error']}")
                        reward = -1
                    else:
                        reward = teacher_response.get("overall_reward", 0)
                        reward_color = "green" if reward == 1 else ("orange" if reward == 0.5 else "red")
                        reward_emoji = "✅" if reward == 1 else ("⚠️" if reward == 0.5 else "❌")
                        
                        st.markdown(f"### {reward_emoji} Reward: `{reward}`")
                        st.markdown(f":{reward_color}[{teacher_response.get('reward_explanation', '')}]")
                        
                        st.markdown("**Advice:**")
                        st.info(teacher_response.get('advice_to_student', ''))
                
                # Train student
                with st.spinner("Updating student model..."):
                    loss = train_student_step(
                        st.session_state.student_model,
                        st.session_state.optimizer,
                        active_image,
                        reward
                    )
                    losses.append(loss)
                    rewards.append(reward)
                    time.sleep(0.2)
                
                st.success(f"Loss: {loss:.4f}")
                
                # Detailed feedback
                if not teacher_response.get("error") and show_step_feedback:
                    st.markdown("---")
                    st.subheader("📝 Step Feedback")
                    
                    step_feedback = teacher_response.get('step_feedback', [])
                    
                    exact = sum(1 for s in step_feedback if s.get('label') == 'EXACT_MATCH')
                    partial = sum(1 for s in step_feedback if s.get('label') == 'PARTIAL')
                    wrong = sum(1 for s in step_feedback if s.get('label') == 'WRONG')
                    
                    cols = st.columns(3)
                    cols[0].metric("✅ Exact", exact)
                    cols[1].metric("⚠️ Partial", partial)
                    cols[2].metric("❌ Wrong", wrong)
                    
                    for step in step_feedback:
                        label = step.get('label', 'UNKNOWN')
                        emoji = "✅" if label == 'EXACT_MATCH' else ("⚠️" if label == 'PARTIAL' else "❌")
                        
                        st.markdown(f"{emoji} **Step {step.get('index')}**: {step.get('student_step', '')}")
                        st.caption(step.get('comment', ''))
                
                if not teacher_response.get("error") and show_teacher_features:
                    st.markdown("---")
                    st.subheader("🎯 Teacher Ground Truth")
                    for i, feat in enumerate(teacher_response.get('teacher_features', []), 1):
                        st.markdown(f"{i}. {feat}")
                
                if show_raw_json and not teacher_response.get("error"):
                    with st.expander("🔍 Raw JSON"):
                        st.json(teacher_response)
            
            progress_bar.progress((iteration + 1) / num_iterations)
        
        status_text.markdown("**✅ Training Complete!**")
        
        # Summary
        st.markdown("---")
        st.header("📈 Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Iterations", num_iterations)
        col2.metric("Avg Reward", f"{sum(rewards)/len(rewards):.2f}")
        col3.metric("Final Reward", rewards[-1])
        col4.metric("Improvement", f"{rewards[-1] - rewards[0]:+.1f}")
        
        # Charts
        import pandas as pd
        
        if show_loss_curve:
            df_loss = pd.DataFrame({
                'Iteration': list(range(1, len(losses) + 1)),
                'Loss': losses
            })
            st.subheader("📉 Training Loss")
            st.line_chart(df_loss.set_index('Iteration'))
        
        df_reward = pd.DataFrame({
            'Iteration': list(range(1, len(rewards) + 1)),
            'Reward': rewards
        })
        st.subheader("📊 Reward Progress")
        st.line_chart(df_reward.set_index('Iteration'))

elif run_button:
    if not active_image:
        st.warning("⚠️ Please draw or upload an image first!")
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 CNN Student Model + Gemma 3 27B Teacher | Built with Streamlit</p>
    <p>Self-Verifiable Visual Feature Learning System</p>
</div>
""", unsafe_allow_html=True)
