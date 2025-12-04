import streamlit as st
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

# Initialize session state
if 'iteration_history' not in st.session_state:
    st.session_state.iteration_history = []
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'student_model' not in st.session_state:
    st.session_state.student_model = None
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'training_active' not in st.session_state:
    st.session_state.training_active = False

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
        model = genai.GenerativeModel('gemma-2-27b-it')
        
        prompt = TEACHER_PROMPT_TEMPLATE.format(
            TASK_NAME=task_name,
            STUDENT_OUTPUT=student_output
        )
        
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
        
        # Simple reward-based loss
        loss = -reward * torch.mean(features)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    except Exception as e:
        return 0.0

# ================================
# MAIN APP FUNCTION
# ================================
def main():
    # Configuration
    api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="Enter your API key")
    
    st.divider()
    
    # Upload image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'webp'])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.uploaded_image = image
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Training Parameters")
        num_iterations = st.number_input("Number of Iterations", 1, 50, 10)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        num_features = st.number_input("Number of Features", 5, 20, 10)
        task_name = st.text_input("Task Name", "Visual Feature Extraction")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Init Model", use_container_width=True):
                st.session_state.student_model, st.session_state.optimizer = initialize_student_model(num_features)
                st.success("Model initialized!")
        
        with col_b:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.iteration_history = []
                st.rerun()
    
    st.divider()
    
    # Display settings
    col1, col2, col3, col4 = st.columns(4)
    show_teacher_features = col1.checkbox("Teacher Features", True)
    show_step_feedback = col2.checkbox("Step Feedback", True)
    show_raw_json = col3.checkbox("Raw JSON", False)
    show_loss_curve = col4.checkbox("Loss Curve", True)
    
    st.divider()
    
    # Run button
    if st.button("▶️ **START TRAINING**", type="primary", use_container_width=True):
        st.session_state.training_active = True
    
    # Training loop
    if st.session_state.training_active and st.session_state.uploaded_image and api_key:
        if st.session_state.student_model is None:
            st.session_state.student_model, st.session_state.optimizer = initialize_student_model(num_features)
        
        st.header("🔄 Training Progress")
        
        progress_bar = st.progress(0)
        status_container = st.container()
        
        losses = []
        rewards = []
        
        for iteration in range(num_iterations):
            with status_container:
                st.subheader(f"Iteration {iteration + 1} / {num_iterations}")
                
                col1, col2 = st.columns(2)
                
                # Student phase
                with col1:
                    st.markdown("### 👨‍🎓 Student CNN")
                    with st.spinner("Analyzing..."):
                        student_output, student_descriptions = call_student_model(
                            st.session_state.uploaded_image, 
                            st.session_state.student_model
                        )
                        time.sleep(0.2)
                    
                    st.code(student_output, language=None)
                
                # Teacher phase
                with col2:
                    st.markdown("### 👨‍🏫 Teacher (Gemma 3)")
                    with st.spinner("Evaluating..."):
                        teacher_response = call_teacher_model(
                            st.session_state.uploaded_image, 
                            student_output, 
                            task_name, 
                            api_key
                        )
                        time.sleep(0.2)
                    
                    if "error" in teacher_response:
                        st.error(f"Error: {teacher_response['error']}")
                        reward = -1
                    else:
                        reward = teacher_response.get("overall_reward", 0)
                        
                        if reward == 1:
                            st.success(f"✅ Reward: **+1**")
                        elif reward == 0.5:
                            st.warning(f"⚠️ Reward: **+0.5**")
                        else:
                            st.error(f"❌ Reward: **-1**")
                        
                        st.info(teacher_response.get('reward_explanation', ''))
                        
                        with st.expander("💡 Advice"):
                            st.write(teacher_response.get('advice_to_student', ''))
                
                # Train student
                with st.spinner("Updating model..."):
                    loss = train_student_step(
                        st.session_state.student_model,
                        st.session_state.optimizer,
                        st.session_state.uploaded_image,
                        reward
                    )
                    losses.append(loss)
                    rewards.append(reward)
                
                st.metric("Training Loss", f"{loss:.4f}")
                
                # Detailed feedback
                if not teacher_response.get("error"):
                    if show_step_feedback:
                        st.markdown("#### 📝 Step-by-Step Feedback")
                        
                        step_feedback = teacher_response.get('step_feedback', [])
                        
                        exact = sum(1 for s in step_feedback if s.get('label') == 'EXACT_MATCH')
                        partial = sum(1 for s in step_feedback if s.get('label') == 'PARTIAL')
                        wrong = sum(1 for s in step_feedback if s.get('label') == 'WRONG')
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("✅ Exact", exact)
                        col_b.metric("⚠️ Partial", partial)
                        col_c.metric("❌ Wrong", wrong)
                        
                        for step in step_feedback:
                            label = step.get('label', 'UNKNOWN')
                            if label == 'EXACT_MATCH':
                                emoji = "✅"
                            elif label == 'PARTIAL':
                                emoji = "⚠️"
                            else:
                                emoji = "❌"
                            
                            st.markdown(f"{emoji} **Step {step.get('index')}**: {step.get('student_step', '')}")
                            st.caption(step.get('comment', ''))
                    
                    if show_teacher_features:
                        st.markdown("#### 🎯 Teacher Ground Truth")
                        for i, feat in enumerate(teacher_response.get('teacher_features', []), 1):
                            st.markdown(f"{i}. {feat}")
                    
                    if show_raw_json:
                        with st.expander("🔍 Raw JSON"):
                            st.json(teacher_response)
                
                st.divider()
            
            progress_bar.progress((iteration + 1) / num_iterations)
        
        st.success("✅ **Training Complete!**")
        
        # Summary
        st.header("📈 Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Iterations", num_iterations)
        col2.metric("Average Reward", f"{sum(rewards)/len(rewards):.2f}")
        col3.metric("Final Reward", f"{rewards[-1]:.1f}")
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
        
        st.session_state.training_active = False
    
    elif st.session_state.training_active:
        if not st.session_state.uploaded_image:
            st.warning("⚠️ Please upload an image first!")
            st.session_state.training_active = False
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key!")
            st.session_state.training_active = False

# Run in canvas mode
if __name__ == "__main__":
    main()
