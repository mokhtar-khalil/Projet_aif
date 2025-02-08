import gradio as gr
from PIL import Image
import requests
import io
import os

API_URL = "http://127.0.0.1:5005/"

def predict_genre(image):
    """
    Predict the genre of a movie poster.
    """
    try:
        # Convert image to binary
        img_binary = io.BytesIO()
        image.save(img_binary, format="JPEG")
        img_binary.seek(0)

        # Call the REST API for prediction
        response = requests.post(f"{API_URL}predict", files={"file": img_binary})
        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def recommend_movies(image):
    """
    Recommend the 5 most similar movies based on a poster.
    """
    try:
        # Convert image to binary
        img_binary = io.BytesIO()
        image.save(img_binary, format="JPEG")
        img_binary.seek(0)

        # Call the REST API for recommendations
        response = requests.post(f"{API_URL}recommend", files={"file": img_binary})
        if response.status_code != 200:
            return [f"Error: {response.text}" for _ in range(5)]

        recommendations = response.json().get("recommendations", [])
        #print("recommendations en f gr", recommendations)

        if not recommendations:
            return ["No recommendations found" for _ in range(5)]

        # Load and return recommended posters
        loaded_images = []
        for rec in recommendations:
            try:
                img = Image.open(os.path.normpath(rec))  # Ensure the file path is correct
                loaded_images.append(img)
            except Exception as e:
                loaded_images.append(f"Error loading {rec}: {str(e)}")

        # Ensure we always return 5 outputs, even if some fail
        while len(loaded_images) < 5:
            loaded_images.append("Error: Missing Recommendation")

        return loaded_images
    except Exception as e:
        return [f"Error: {str(e)}" for _ in range(5)]

def recommend_by_description(description, method):
    """
    Recommend movies based on their plot description and selected embedding method.
    """
    try:
        response = requests.post(
            f"{API_URL}recommend_by_description",
            json={"description": description, "method": method}
        )
        recommendations = response.json().get("recommendations", []) if response.status_code == 200 else []
        return "\n".join([rec["title"] for rec in recommendations]) if recommendations else "No recommendations found"
    except Exception as e:
        return f"Error: {str(e)}"



# Separate interfaces for each function
predict_interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil", label="Upload Movie Poster for Genre Prediction"),
    outputs=gr.Textbox(label="Predicted Genre"),
    title="Movie Genre Predictor",
    description="Upload a movie poster to predict its genre."
)

recommend_interface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Image(type="pil", label="Upload Movie Poster for Recommendations"),
    outputs=[gr.Image(label=f"Recommendation {i+1}") for i in range(5)],
    title="Movie Recommender System",
    description="Upload a movie poster to get recommendations for similar movies."
)

recommend_by_text_interface = gr.Interface(
    fn=recommend_by_description,
    inputs=[
        gr.Textbox(label="Enter Movie Description"),
        gr.Radio(["tfidf", "glove", "bert"], label="Choose Embedding Method", value="tfidf")
    ],
    outputs=gr.Textbox(label="Recommended Movies"),
    title="Movie Recommender System (by Description)"
)

# Combine interfaces into tabs
combined_interface = gr.TabbedInterface(
    [predict_interface, recommend_interface, recommend_by_text_interface], 
    ["Predict Genre", "Recommend by Poster", "Recommend by Description"]
)
if __name__ == "__main__":
    try:
        print("Starting Gradio app...")
        port = int(os.getenv("GRADIO_SERVER_PORT", 7860)) 
        combined_interface.launch(server_name="127.0.0.1", server_port=port)
    except Exception as e:
        print(f"App crashed: {e}")
