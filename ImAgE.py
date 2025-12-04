import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import io
import time

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

# Sidebar controls
st.sidebar.title("⚙️ Configuration")

# API Key
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")

st.sidebar.markdown("---")

# Parameters
st.sidebar.subheader("Training Parameters")
num_iterations = st.sidebar.slider("Number of Iterations", 1, 20, 5, help="How many student-teacher cycles to run")
temperature = st.sidebar.slider("Student Temperature", 0.0, 2.0, 0.7, 0.1, help="Higher = more creative, Lower = more deterministic")
max_steps = st.sidebar.number_input("Max Features", 5, 20, 10, help="Number of visual features to extract")
task_name = st.sidebar.text_input("Task Name", "Visual Feature Extraction", help="Name of the current task")

st.sidebar.markdown("---")

# Display settings
st.sidebar.subheader("Display Settings")
show_teacher_features = st.sidebar.checkbox("Show Teacher's Ground Truth", True)
show_step_feedback = st.sidebar.checkbox("Show Step-by-Step Feedback", True)
show_raw_json = st.sidebar.checkbox("Show Raw JSON Response", False)
auto_scroll = st.sidebar.checkbox("Auto-scroll to Latest", True)

st.sidebar.markdown("---")

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    run_button = st.button("▶️ Run", type="primary", use_container_width=True)
with col2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.session_state.iteration_history = []
    st.session_state.current_iteration = 0
    st.rerun()

# Main title
st.title("🎓 Student-Teacher Visual Feature Learning System")
st.markdown("*Inspired by DeepSeekMath-V2's self-verifiable reasoning approach*")

# Image upload
uploaded_file = st.file_uploader("📷 Upload an image", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    with col2:
        st.info(f"""
        **Image Info:**
        - Size: {image.size[0]} x {image.size[1]} pixels
        - Format: {image.format}
        - Mode: {image.mode}
        """)

# Teacher prompt template
TEACHER_PROMPT_TEMPLATE = """You are an expert TEACHER model supervising a STUDENT model in a self-training loop.
Your job is to:
1. Look at the input image.
2. Infer the 10 most important basic visual features of that image.
3. Compare those features and reasoning steps with the STUDENT's attempt.
4. Provide strict, step-by-step feedback and a numeric reward.

--------------------
ROLE & GOAL
--------------------
- You are the TEACHER (expert).
- The STUDENT is a weaker model that:
  - Sees the same image.
  - Predicts 10 basic visual features.
  - Writes step-by-step reasoning for how it derived those features.
- Your goal is to:
  - Check every STUDENT step.
  - Identify any incorrect or logically broken step.
  - Assign a global reward: +1, 0.5, or -1.

--------------------
TASK DEFINITION
--------------------
Current task name: {TASK_NAME}

For this task, a "basic visual feature" can include:
- Main objects present.
- Dominant colors.
- Overall layout or composition (where things are).
- Background characteristics.
- Lighting / shadows.
- Texture or style (e.g., cartoon, photo, sketch).
- Presence of text or symbols.
- Any other simple, stable property directly visible in the image.

You must infer **exactly 10** such basic visual features that are:
- High-level, easy to learn,
- Directly grounded in the image,
- Non-overlapping as much as possible.

--------------------
INPUTS
--------------------
1. IMAGE INPUT:
   - You receive the raw image via the API (multimodal input).

2. STUDENT OUTPUT:
   The Student's answer will be given between special tags:

   <STUDENT_OUTPUT_BEGIN>
   {STUDENT_OUTPUT}
   <STUDENT_OUTPUT_END>

The student output MAY include:
- A numbered list of 10 predicted features.
- Step-by-step reasoning or explanation for each feature.
- Intermediate thoughts.

You MUST treat this as a fallible attempt that can contain:
- Missing features
- Hallucinated features (not in the image)
- Incorrect reasoning
- Inconsistent or vague statements

--------------------
WHAT YOU MUST DO
--------------------
Your evaluation process MUST follow these phases INTERNALLY
(you do NOT print these internal thoughts):

Phase 1 — Teacher's own understanding:
- Carefully inspect the image.
- Infer the 10 best "ground-truth" basic visual features.
- These are your TEACHER_FEATURES.

Phase 2 — Align Student features against Teacher features:
- Parse the student output into a list of features and steps.
- For each student feature / step:
  - Decide if it is:
    - EXACT_MATCH: Correct and clearly grounded in the image.
    - PARTIAL: Mostly correct but missing detail or slightly inaccurate.
    - WRONG: Not supported by the image, clearly incorrect, or logically broken.
  - Add a short comment explaining why.

Phase 3 — Global reward:
- Let N = number of student steps you evaluate.
- Let C = number of steps that are EXACT_MATCH or clearly correct.
- Let W = number of WRONG steps.
- Let P = number of PARTIAL steps.

Assign the global reward as follows:

- Reward = +1
  - All evaluated steps are correct or at most very minor issues.
  - No clearly WRONG or hallucinated features.
  - Reasoning is logically sound.

- Reward = 0.5
  - More than half of the steps are correct.
  - Some PARTIAL or slightly flawed reasoning, but overall usable.
  - At most 1–2 WRONG steps and they do not break the full understanding.

- Reward = -1
  - Any of the following:
    - The majority of steps are WRONG.
    - Critical hallucinations about objects or colors.
    - Reasoning is mostly inconsistent or impossible to trust.

You MUST choose EXACTLY one of: 1, 0.5, -1.

--------------------
OUTPUT FORMAT (VERY IMPORTANT)
--------------------
You MUST output ONLY a single JSON object with the following structure:

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
      "student_step": "raw text of the student's first feature / step",
      "label": "EXACT_MATCH | PARTIAL | WRONG",
      "comment": "Short explanation in 1-3 sentences."
    }},
    {{
      "index": 2,
      "student_step": "...",
      "label": "EXACT_MATCH | PARTIAL | WRONG",
      "comment": "..."
    }}
  ],
  "overall_reward": -1,
  "reward_explanation": "1-3 sentences explaining why this reward was chosen.",
  "advice_to_student": "2-5 sentences giving actionable guidance to improve next time."
}}

Rules:
- "teacher_features" MUST have exactly 10 items.
- "overall_reward" MUST be one of: -1, 0.5, 1 (as a number, not a string).
- Do NOT include any extra keys.
- Do NOT output any text before or after the JSON.
- Do NOT reveal your internal reasoning or chain-of-thought. Only give the JSON.
"""

STUDENT_PROMPT_TEMPLATE = """You are a STUDENT model learning to extract visual features from images.

Your task is to:
1. Carefully observe the provided image.
2. Identify exactly {MAX_STEPS} basic visual features.
3. For each feature, explain your reasoning step-by-step.

A "basic visual feature" can include:
- Main objects present
- Dominant colors
- Overall layout or composition
- Background characteristics
- Lighting and shadows
- Texture or style
- Presence of text or symbols
- Any other simple, stable property directly visible

OUTPUT FORMAT:
Please structure your response as a numbered list:

1. [Feature name]: [Brief description]
   Reasoning: [Explain how you identified this feature]

2. [Feature name]: [Brief description]
   Reasoning: [Explain how you identified this feature]

... and so on for all {MAX_STEPS} features.

Be specific, accurate, and ground every feature in what you actually see in the image.
"""

def call_student_model(image, iteration_num, temperature, max_steps):
    """Simulate student model using Gemini with lower capability"""
    try:
        genai.configure(api_key=api_key)
        
        # Use a slightly weaker configuration for student
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = STUDENT_PROMPT_TEMPLATE.format(MAX_STEPS=max_steps)
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error in student model: {str(e)}"

def call_teacher_model(image, student_output, task_name, max_steps):
    """Call Gemini 2.5 Flash as teacher/verifier"""
    try:
        genai.configure(api_key=api_key)
        
        # Use the most capable model for teacher
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = TEACHER_PROMPT_TEMPLATE.format(
            TASK_NAME=task_name,
            STUDENT_OUTPUT=student_output,
            MAX_STEPS=max_steps
        )
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent evaluation
            )
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        return json.loads(response_text)
    except Exception as e:
        return {"error": str(e)}

# Run training iterations
if run_button and uploaded_file and api_key:
    st.markdown("---")
    st.header("🔄 Training Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(num_iterations):
        status_text.markdown(f"**Iteration {iteration + 1} / {num_iterations}**")
        
        with st.expander(f"📊 Iteration {iteration + 1}", expanded=(iteration == num_iterations - 1)):
            col1, col2 = st.columns(2)
            
            # Student phase
            with col1:
                st.subheader("👨‍🎓 Student Output")
                with st.spinner("Student analyzing image..."):
                    student_output = call_student_model(image, iteration, temperature, max_steps)
                    time.sleep(0.5)  # Brief delay for UI
                
                st.markdown("**Student's Analysis:**")
                st.text_area(
                    "Student Response",
                    student_output,
                    height=300,
                    key=f"student_{iteration}",
                    label_visibility="collapsed"
                )
            
            # Teacher phase
            with col2:
                st.subheader("👨‍🏫 Teacher Evaluation")
                with st.spinner("Teacher evaluating..."):
                    teacher_response = call_teacher_model(image, student_output, task_name, max_steps)
                    time.sleep(0.5)  # Brief delay for UI
                
                if "error" in teacher_response:
                    st.error(f"Teacher Error: {teacher_response['error']}")
                else:
                    # Display reward prominently
                    reward = teacher_response.get("overall_reward", 0)
                    reward_color = "green" if reward == 1 else ("orange" if reward == 0.5 else "red")
                    reward_emoji = "✅" if reward == 1 else ("⚠️" if reward == 0.5 else "❌")
                    
                    st.markdown(f"### {reward_emoji} Reward: `{reward}`")
                    st.markdown(f":{reward_color}[{teacher_response.get('reward_explanation', 'No explanation')}]")
                    
                    st.markdown("**Advice:**")
                    st.info(teacher_response.get('advice_to_student', 'No advice'))
            
            # Detailed feedback section
            if not teacher_response.get("error") and show_step_feedback:
                st.markdown("---")
                st.subheader("📝 Step-by-Step Feedback")
                
                step_feedback = teacher_response.get('step_feedback', [])
                
                # Create metrics
                exact_match = sum(1 for s in step_feedback if s.get('label') == 'EXACT_MATCH')
                partial = sum(1 for s in step_feedback if s.get('label') == 'PARTIAL')
                wrong = sum(1 for s in step_feedback if s.get('label') == 'WRONG')
                
                metric_cols = st.columns(3)
                metric_cols[0].metric("✅ Exact Match", exact_match)
                metric_cols[1].metric("⚠️ Partial", partial)
                metric_cols[2].metric("❌ Wrong", wrong)
                
                # Display feedback for each step
                for step in step_feedback:
                    label = step.get('label', 'UNKNOWN')
                    emoji = "✅" if label == 'EXACT_MATCH' else ("⚠️" if label == 'PARTIAL' else "❌")
                    color = "green" if label == 'EXACT_MATCH' else ("orange" if label == 'PARTIAL' else "red")
                    
                    with st.container():
                        st.markdown(f"**{emoji} Step {step.get('index', '?')}**: {label}")
                        st.markdown(f"*Student:* {step.get('student_step', 'N/A')}")
                        st.markdown(f":{color}[{step.get('comment', 'No comment')}]")
                        st.markdown("")
            
            # Teacher's ground truth
            if not teacher_response.get("error") and show_teacher_features:
                st.markdown("---")
                st.subheader("🎯 Teacher's Ground Truth Features")
                teacher_features = teacher_response.get('teacher_features', [])
                for i, feature in enumerate(teacher_features, 1):
                    st.markdown(f"{i}. {feature}")
            
            # Raw JSON
            if show_raw_json and not teacher_response.get("error"):
                st.markdown("---")
                with st.expander("🔍 Raw JSON Response"):
                    st.json(teacher_response)
            
            # Store in history
            st.session_state.iteration_history.append({
                'iteration': iteration + 1,
                'student_output': student_output,
                'teacher_response': teacher_response,
                'reward': teacher_response.get('overall_reward', 0) if 'error' not in teacher_response else -1
            })
        
        progress_bar.progress((iteration + 1) / num_iterations)
    
    status_text.markdown("**✅ Training Complete!**")
    
    # Summary statistics
    st.markdown("---")
    st.header("📈 Training Summary")
    
    if st.session_state.iteration_history:
        rewards = [h['reward'] for h in st.session_state.iteration_history]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Iterations", len(rewards))
        col2.metric("Average Reward", f"{sum(rewards) / len(rewards):.2f}")
        col3.metric("Best Reward", max(rewards))
        col4.metric("Improvement", f"{rewards[-1] - rewards[0]:+.1f}")
        
        # Reward chart
        import pandas as pd
        
        df = pd.DataFrame({
            'Iteration': list(range(1, len(rewards) + 1)),
            'Reward': rewards
        })
        
        st.line_chart(df.set_index('Iteration'))
        
        # Download results
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(st.session_state.iteration_history, indent=2),
            file_name=f"training_results_{int(time.time())}.json",
            mime="application/json"
        )

elif run_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload an image first!")
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit + Google Gemini | Inspired by DeepSeekMath-V2</p>
    <p>Student-Teacher Self-Verifiable Learning System</p>
</div>
""", unsafe_allow_html=True)
