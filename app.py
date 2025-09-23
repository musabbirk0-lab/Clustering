import gradio as gr
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Prediction function
def predict(x, y):
    point = np.array([[x, y]])
    cluster = kmeans.predict(point)[0]
    return f"This point belongs to cluster {cluster}"

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="X coordinate"), gr.Number(label="Y coordinate")],
    outputs=gr.Textbox(label="Cluster"),
    title="K-Means Clustering Demo",
    description="Enter X and Y coordinates to see which cluster the point belongs to."
)

if __name__ == "__main__":
    demo.launch()
