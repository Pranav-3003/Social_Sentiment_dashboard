import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from src.utils.db import DBManager, Post
from src.features.vectorizers import TextVectorizer
from src.models.classical_ml import ClassicalMLWorkbench
from src.models.deep_learning import KerasDLWorkbench
from src.models.transformers_model import TransformerWorkbench
from textblob import TextBlob

def render_workbench(db_manager: DBManager):
    st.markdown("## Machine Learning Workbench")
    st.write("Train, fine-tune, and evaluate models to classify sentiment polarity.")
    
    with st.expander("📚 Model Workbench Guide: How to choose and train models", expanded=False):
        st.markdown("""
        Choose the model type that fits your speed, accuracy, and interpretability requirements:
        
        - 🚀 **Classical Machine Learning (CPU)**: Models like Logistic Regression, Support Vector Machines (SVM), and Random Forests train instantly on CPU. They are highly interpretable, especially when combined with SHAP and LIME word-level contribution chips.
        - 🧠 **Deep Learning (TensorFlow/Keras)**: Multi-layer neural networks like LSTMs or CNNs. They process texts as word sequences, capturing basic sequential context.
        - ⚡ **Transformer Models (GPU/CPU)**: High-capacity architectures like BERT, RoBERTa, and DistilBERT. They utilize self-attention mechanisms for state-of-the-art accuracy, but require more computing power.
        """)
    
    session = db_manager.get_session()
    posts = session.query(Post).all()
    session.close()
    
    if not posts:
        st.warning("No data found in the database. Ingest and Clean a dataset first before training models!")
        return

    # Prepare DataFrame
    df = pd.DataFrame([{
        "text": p.cleaned_text or p.text,
        "raw_text": p.text
    } for p in posts])
    
    # Auto-label using TextBlob for workbench training target
    labels = []
    sentiment_scores = []
    for text in df['text']:
        score = TextBlob(text).sentiment.polarity
        sentiment_scores.append(score)
        if score > 0.05:
            labels.append(2) # Positive
        elif score < -0.05:
            labels.append(0) # Negative
        else:
            labels.append(1) # Neutral
            
    df['label'] = labels
    df['sentiment_score'] = sentiment_scores
    
    tab1, tab2, tab3 = st.tabs(["Classical Machine Learning", "Deep Learning Networks", "Transformer Models"])

    # TAB 1: Classical Machine Learning
    with tab1:
        st.markdown("### Classical ML Benchmarks")
        
        # 1. Feature Representation
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            vect_method = st.selectbox("Text Vectorization Method", ["tfidf", "count"])
        with col_f2:
            max_feats = st.slider("Max Vectorizer Features", 500, 10000, 3000, step=500)
            
        # 2. Train-Test Split
        split_ratio = st.slider("Train-Test Split Ratio", 0.5, 0.9, 0.8, step=0.05)
        
        # 3. Model Selector
        workbench = ClassicalMLWorkbench()
        available_models = workbench.get_available_models()
        selected_model = st.selectbox("Select ML Classifier", available_models)
        
        # Hyperparameters
        st.markdown("#### Hyperparameters & Tuning")
        col_h1, col_h2 = st.columns(2)
        
        tune_hp = st.checkbox("Enable Grid Search Hyperparameter Tuning")
        
        params = {}
        param_grid = {}
        
        if selected_model == "Logistic Regression":
            with col_h1:
                c_val = st.number_input("C (Regularization)", 0.01, 10.0, 1.0, help="Inverse of regularization strength; smaller values specify stronger regularization.")
                params = {"C": c_val}
            with col_h2:
                if tune_hp:
                    param_grid = {"C": [0.1, 1.0, 10.0]}
                    
        elif selected_model == "Naive Bayes":
            with col_h1:
                alpha = st.number_input("Alpha (Smoothing)", 0.01, 5.0, 1.0, help="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).")
                params = {"alpha": alpha}
            with col_h2:
                if tune_hp:
                    param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0]}
                    
        elif selected_model == "Random Forest":
            with col_h1:
                n_est = st.slider("N Estimators", 10, 300, 100, help="The number of decision trees in the forest.")
                max_d = st.slider("Max Depth", 2, 30, 10, help="The maximum depth of the trees. None represents unlimited depth.")
                params = {"n_estimators": n_est, "max_depth": max_d}
            with col_h2:
                if tune_hp:
                    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
                    
        elif selected_model == "SVM":
            with col_h1:
                c_val = st.number_input("C (Cost)", 0.01, 10.0, 1.0, help="Regularization parameter. The strength of the regularization is inversely proportional to C.")
                params = {"C": c_val}
            with col_h2:
                if tune_hp:
                    param_grid = {"C": [0.1, 1.0, 5.0]}
                    
        elif selected_model in ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
            with col_h1:
                n_est = st.slider("Estimators", 10, 200, 100, help="The number of sequential trees to build.")
                lr = st.number_input("Learning Rate", 0.01, 1.0, 0.1, help="Step size shrinkage used in update to prevent overfitting.")
                params = {"n_estimators": n_est, "learning_rate": lr}
            with col_h2:
                if tune_hp:
                    param_grid = {"learning_rate": [0.05, 0.1, 0.2]}

        # Trigger Train
        if st.button(f"Train {selected_model}", key="btn_train_classical"):
            with st.spinner(f"Vectorizing texts and training {selected_model}..."):
                # Fit vectorizer
                vectorizer = TextVectorizer(method=vect_method, max_features=max_feats)
                X = vectorizer.fit_transform(df['text'].tolist())
                y = df['label'].values
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=split_ratio, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Fit model
                model, cv_score = workbench.train(
                    model_name=selected_model,
                    X_train=X_train,
                    y_train=y_train,
                    params=params,
                    tune_hyperparameters=tune_hp,
                    param_grid=param_grid,
                    cv=3
                )
                
                # Save state to session
                st.session_state[f"model_{selected_model}"] = model
                st.session_state["active_model"] = model
                st.session_state["active_model_name"] = selected_model
                st.session_state["active_vectorizer"] = vectorizer
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Evaluation Metrics
                acc = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                
                st.success(f"{selected_model} successfully trained!")
                
                # 4. Metrics Cards
                st.markdown("#### Evaluation Benchmarks")
                cols_met = st.columns(4)
                cols_met[0].metric("Accuracy Score", f"{acc:.4f}")
                cols_met[1].metric("Weighted Precision", f"{precision:.4f}")
                cols_met[2].metric("Weighted Recall", f"{recall:.4f}")
                cols_met[3].metric("Weighted F1-Score", f"{f1:.4f}")
                
                # Cross validation score
                st.info(f"3-Fold Cross Validation Mean Train Accuracy: {cv_score:.4f}")
                
                # Plots: Confusion Matrix, ROC, Learning Curves, Feature Importance
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    labels_text = ["Negative", "Neutral", "Positive"]
                    # Render Heatmap
                    fig_cm = px.imshow(
                        cm, x=labels_text[:cm.shape[1]], y=labels_text[:cm.shape[0]],
                        text_auto=True, color_continuous_scale="Viridis",
                        title="Testing Confusion Matrix", template="plotly_dark"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with col_plot2:
                    st.write("#### Learning Curve")
                    try:
                        tr_sizes, tr_scores, val_scores = workbench.get_learning_curve_data(selected_model, X_train, y_train, cv=3)
                        lc_df = pd.DataFrame({
                            "Training Size": tr_sizes,
                            "Train Score": tr_scores,
                            "Validation Score": val_scores
                        })
                        fig_lc = px.line(lc_df, x="Training Size", y=["Train Score", "Validation Score"],
                                         title="Model Convergence Curve", template="plotly_dark")
                        st.plotly_chart(fig_lc, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not build learning curve: {e}")

                # Feature Importance
                st.write("---")
                st.write("#### Word Feature Importance / Weights Analysis")
                try:
                    feat_names = list(vectorizer.vectorizer.get_feature_names_out())
                    imp_df = workbench.get_feature_importances(selected_model, feat_names).head(15)
                    
                    if not imp_df.empty:
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                         color="Importance", title="Top 15 Predictive Words", template="plotly_dark")
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("Feature importance not supported for this model.")
                except Exception as e:
                    st.error(f"Error computing feature importance: {e}")

    # TAB 2: Deep Learning Networks
    with tab2:
        st.markdown("### Deep Learning Workbench")
        st.write("Train Recurrent and Convolutional Neural Networks on social media sequence tokens.")
        
        dl_model = st.selectbox("Select Deep Learning Architecture", ["Simple Neural Network", "LSTM", "Bi-LSTM", "GRU", "CNN for text"])
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            epochs = st.slider("Training Epochs", 1, 20, 5)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with col_d2:
            emb_dim = st.slider("Word Embedding Dimensions", 50, 300, 100, step=50)
            max_seq_len = st.slider("Max Sequence Tokens Padding", 30, 200, 100, step=10)
            
        if st.button(f"Train DL {dl_model}", key="btn_train_dl"):
            with st.spinner(f"Tokenizing and training {dl_model}..."):
                # Vectorize text as integer sequences
                vectorizer = TextVectorizer(method='word2vec')
                sequences = vectorizer.fit_transform(df['text'].tolist())
                vocab_size = len(vectorizer.word2idx)
                y = df['label'].values
                
                # Train/test split
                seq_train, seq_test, y_train, y_test = train_test_split(
                    sequences, y, train_size=split_ratio, random_state=42
                )
                
                dl_workbench = KerasDLWorkbench(max_len=max_seq_len)
                
                # Fit
                num_classes = 3
                model, history = dl_workbench.train(
                    model_name=dl_model,
                    X_train_seq=seq_train,
                    y_train=y_train,
                    vocab_size=vocab_size,
                    num_classes=num_classes,
                    epochs=epochs,
                    batch_size=batch_size,
                    embedding_dim=emb_dim
                )
                
                # Save state
                st.session_state[f"model_{dl_model}"] = dl_workbench
                st.session_state["active_model"] = dl_workbench
                st.session_state["active_model_name"] = dl_model
                st.session_state["active_vectorizer"] = vectorizer
                
                # Plot History
                st.success(f"Deep learning network {dl_model} trained successfully!")
                
                hist_df = pd.DataFrame(history)
                fig_loss = px.line(hist_df, y=['accuracy', 'val_accuracy'], title="Accuracy Progress per Epoch", template="plotly_dark")
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Evaluate on test set
                y_pred = dl_workbench.predict(dl_model, seq_test)
                acc = accuracy_score(y_test, y_pred)
                
                st.metric("Test Set Accuracy", f"{acc:.4f}")

    # TAB 3: Transformer Models
    with tab3:
        st.markdown("### Transformers workbench")
        st.write("Fine-tune high-capacity models or load pre-trained classifiers for zero-shot predictions.")
        
        selected_trans = st.selectbox("Select Transformer Base Model", ["DistilBERT", "BERT", "RoBERTa", "DeBERTa"])
        
        t_epochs = st.slider("Fine-Tuning Epochs (Transformers)", 1, 5, 1)
        t_lr = st.selectbox("Learning Rate", [1e-5, 2e-5, 5e-5], index=1)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("Load Pre-trained Pipeline", help="Load the model weights directly for prediction (fast, no training needed)"):
                with st.spinner("Downloading and loading transformer model pipeline..."):
                    trans_wb = TransformerWorkbench()
                    success = trans_wb.load_pipeline(selected_trans)
                    if success:
                        st.session_state["active_model"] = trans_wb
                        st.session_state["active_model_name"] = f"Pretrained-{selected_trans}"
                        st.session_state["active_vectorizer"] = "transformer"
                        st.success(f"{selected_trans} pipeline loaded and activated successfully!")
                    else:
                        st.error("Failed to load transformer model.")
                        
        with col_t2:
            if st.button("Fine-tune on Ingested Corpus", help="Run local backpropagation to fit model weights on your dataset"):
                with st.spinner("Starting transformer backpropagation training loop..."):
                    trans_wb = TransformerWorkbench(max_len=64)
                    y = df['label'].values
                    res = trans_wb.fine_tune(
                        model_display_name=selected_trans,
                        texts=df['text'].tolist(),
                        labels=y,
                        epochs=t_epochs,
                        learning_rate=t_lr
                    )
                    
                    if res["status"] == "success":
                        st.session_state["active_model"] = trans_wb
                        st.session_state["active_model_name"] = f"FineTuned-{selected_trans}"
                        st.session_state["active_vectorizer"] = "transformer"
                        st.success("Fine-tuning completed successfully!")
                        st.write("Accuracy progress:", res["accuracy"])
                    else:
                        st.error(f"Fine-tuning failed: {res['message']}")
                        st.info("Ensure the HuggingFace transformers package is fully installed and your environment supports TensorFlow execution.")
