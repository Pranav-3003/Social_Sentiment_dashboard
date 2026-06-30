import streamlit as st
import streamlit.components.v1 as components
from src.explainability.xai_explain import ExplainabilityWorkbench
from src.utils.db import DBManager

def render_explain(db_manager: DBManager):
    st.markdown("## Explainable AI (XAI)")
    st.write("Understand model predictions by looking at word-level feature attributions.")
    
    active_model_name = st.session_state.get("active_model_name", None)
    active_model = st.session_state.get("active_model", None)
    active_vect = st.session_state.get("active_vectorizer", None)
    
    if active_model is None or active_vect is None or active_vect == "transformer":
        st.warning(
            "XAI requires a custom trained classical ML model (e.g. SVM or Logistic Regression). "
            "Please go to the **Model Workbench** page to train a model first! "
            "A fallback mock explanation is shown below for demo purposes."
        )
        # Setup mock active model/vectorizer for demo
        from sklearn.linear_model import LogisticRegression
        from src.features.vectorizers import TextVectorizer
        
        # Fit dummy vectorizer and model
        demo_vect = TextVectorizer(method="tfidf")
        demo_vect.fit(["good product", "terrible service", "awesome battery", "it crashes constantly"])
        demo_model = LogisticRegression()
        demo_model.fit(demo_vect.transform(["good product", "terrible service", "awesome battery", "it crashes constantly"]), [2, 0, 2, 0])
        
        active_model = demo_model
        active_vect = demo_vect
        active_model_name = "Logistic Regression (Demo Model)"

    st.info(f"**Explaining Predictions of Model:** {active_model_name}")

    explain_text = st.text_area(
        "Enter document to explain", 
        value="This product is absolutely amazing! However, the delivery was late and the support agent was rude."
    )
    
    xai_method = st.radio("Explainability Framework", ["SHAP (Linguistic Perturbation)", "LIME (Local Interpretable Model-agnostic Explanations)"], horizontal=True)
    
    xai_wb = ExplainabilityWorkbench()

    if st.button("Generate Explanation Graph", type="primary", key="btn_run_xai"):
        if explain_text.strip():
            with st.spinner("Generating word attribution scores..."):
                
                # 1. SHAP Explainer
                if "SHAP" in xai_method:
                    pos_contribs, neg_contribs, confidence = xai_wb.explain_with_shap(
                        text=explain_text,
                        model=active_model,
                        vectorizer=active_vect,
                        class_names=["Negative", "Neutral", "Positive"]
                    )
                    
                    st.write("---")
                    st.markdown("### SHAP Word Attribution Chips")
                    st.markdown("""
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: 10px; border: 1px solid rgba(0, 242, 254, 0.15); margin-bottom: 20px;'>
                        <h5 style='margin-top:0; color:#00f2fe;'>💡 SHAP Interpretation Guide</h5>
                        <p style='font-size:0.9rem; margin-bottom:10px;'>SHAP (SHapley Additive exPlanations) values measure each word's individual contribution to the model's output compared to a baseline prediction.</p>
                        <ul style='margin-bottom:0; font-size:0.9rem; padding-left:20px;'>
                            <li>🟢 <span style='color: #4ade80; font-weight: bold;'>Positive (+ score)</span>: Words highlighted in green <b>increase the probability</b> of the predicted class.</li>
                            <li>🔴 <span style='color: #f87171; font-weight: bold;'>Negative (- score)</span>: Words highlighted in red <b>decrease the probability</b> of the predicted class.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence
                    st.markdown(f"**Prediction Confidence Score:** `{confidence:.2f}`")
                    
                    # Render chips
                    chips_html = xai_wb.generate_html_explanation_chips(pos_contribs, neg_contribs)
                    st.markdown(chips_html, unsafe_allow_html=True)
                    
                    # Create detail list
                    st.write("#### Exact Word Impact Table")
                    details = []
                    for w, val in pos_contribs:
                        details.append({"Word": w, "Impact": val, "Direction": "Positive (+)"})
                    for w, val in neg_contribs:
                        details.append({"Word": w, "Impact": val, "Direction": "Negative (-)"})
                        
                    if details:
                        st.dataframe(pd.DataFrame(details))
                    else:
                        st.info("No single word showed significant standalone prediction attribution.")
                
                # 2. LIME Explainer
                else:
                    html_rep, word_weights, pred_class = xai_wb.explain_with_lime(
                        text=explain_text,
                        model=active_model,
                        vectorizer=active_vect,
                        class_names=["Negative", "Neutral", "Positive"],
                        num_features=10
                    )
                    
                    st.write("---")
                    st.markdown("### LIME Feature Attribution Report")
                    st.write(f"**Predicted Sentiment Class:** `{['Negative', 'Neutral', 'Positive'][pred_class]}`")
                    
                    st.markdown("""
                    <div style='background: rgba(15, 23, 42, 0.5); padding: 16px; border-radius: 10px; border: 1px solid rgba(0, 242, 254, 0.15); margin-bottom: 20px;'>
                        <h5 style='margin-top:0; color:#00f2fe;'>💡 LIME Interpretation Guide</h5>
                        <p style='font-size:0.9rem; margin-bottom:0;'>LIME (Local Interpretable Model-agnostic Explanations) builds a local surrogate model around this specific post to see how perturbations (dropping words) affect classifications. The chart below shows the top words influencing this decision locally.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Render LIME HTML representation inside a safe Streamlit iframe component
                    components.html(html_rep, height=500, scrolling=True)
                    
                    # Display text table
                    st.write("#### LIME Weight Scores")
                    lime_df = pd.DataFrame(word_weights, columns=["Feature Word", "Local Attribution Score"])
                    st.dataframe(lime_df)
        else:
            st.error("Please enter a non-empty text string.")
