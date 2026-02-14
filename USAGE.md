# How to Run AI Breast Cancer Detection

## Prerequisites
1.  **Activate Virtual Environment**:
    -   Windows: `.\.venv\Scripts\activate`
    -   Mac/Linux: `source .venv/bin/activate`
2.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

## Running the Backend
1.  Open a terminal in the project root.
2.  Activate the virtual environment (if not already active).
3.  Run the Flask server:
    ```bash
    python backend/app.py
    ```
    You should see output indicating the server is running on `http://0.0.0.0:5000`.

## Using the Feature
1.  Open `index.html` in your browser.
    -   You can just double-click the file, or serve it using a simple HTTP server (recommended for better compatibility):
        ```bash
        python -m http.server 8000
        ```
        Then go to `http://localhost:8000`.
2.  Scroll down to the **AI Breast Cancer Detection** section.
3.  Upload a **CC View** image and an **MLO View** image.
4.  Click **Analyze Risk**.

## Verification
To verify the backend is working without the UI, keep the server running and in another terminal (with venv activated) run:
```bash
python backend/verify.py
```
This will send test images to the local server and print the response.

