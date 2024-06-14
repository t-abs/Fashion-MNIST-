import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from matplotlib.colors import ListedColormap

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Combine train and test sets
images = np.concatenate((x_train, x_test))
labels = np.concatenate((y_train, y_test))

# Define class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to display fashion images
def display_fashion_image(seed):
    np.random.seed(seed)
    random_index = np.random.randint(0, len(images))  # Select a random image
    random_image = images[random_index]
    random_label = labels[random_index]
    return random_image, class_names[random_label]

# Streamlit interface
st.title("Colorful Fashion Design Viewer using Fashion MNIST")

# User input for the random seed
seed = st.number_input("Enter a seed value:", value=0, min_value=0, step=1)

# Generate image button
if st.button("Generate Fashion Design"):
    generated_image, label = display_fashion_image(seed)
    
    # Apply a colormap to make the image colorful
    cmap = plt.cm.viridis  # Choose a colormap
    colored_image = cmap(generated_image / 255.0)

    # Display the generated image
    st.write(f"Fashion Item: {label}")
    fig, ax = plt.subplots()
    ax.imshow(colored_image)
    ax.axis('off')
    st.pyplot(fig)
