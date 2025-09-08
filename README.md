# Healthcare Diagnostic Chatbot using RNN & Flask

A web-based diagnostic chatbot that processes user-described symptoms and engages in a conversational Q&A to suggest possible conditions. Built with a custom Recurrent Neural Network (RNN) model and a Flask REST API backend.

Features

- **Conversational Interface**: Interactive chat interface that asks follow-up questions based on initial symptoms.
- **RNN-Powered Engine**: A Recurrent Neural Network model trained from scratch to understand symptom-disease relationships.
- **RESTful API**: A Flask-based backend with well-defined endpoints for chat functionality.
- **MySQL Integration**: Efficient storage and retrieval of chat logs and user interactions.
- **Modular OOP Design**: Codebase structured using Object-Oriented Principles for maintainability and scalability.

Tech Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** PyTorch, NumPy, Pandas
*   **Database:** MySQL
*   **Frontend (Optional):** HTML, CSS, JavaScript (if you have a simple UI)
*   **Version Control:** Git, GitHub

Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/healthcare-chatbot.git
    cd healthcare-chatbot
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the MySQL database:**
    *   Create a database named `chatbot_db`.
    *   Update the database connection URI in `app/__init__.py` or a config file with your credentials.

5.  **Run the Flask application:**
    ```bash
    python run.py
    ```
    The app will be running at `http://localhost:5000`.

Model Training (If Applicable)

The RNN model was trained on a dataset of symptoms and diseases.
1.  The training script is located in `training/train.py`.
2.  The model weights are saved to `model.pth` (add this file to `.gitignore` if it's large).

