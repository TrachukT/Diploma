# Diploma


# Project README

## Installation

### Prerequisites

1. **Python 3.x** (Recommended version: 3.11)

2. Install all necessary dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

   This will install **FastAPI** and other libraries, including the ones needed for your project to run correctly.

## Configuration

### Step 1: Set the Class Type and Detection Type

- Open `config.py`.
- Modify the variables related to **classes** and **detection type**. These variables define the configuration for the application, so make sure to set them according to your use case.

### Step 2: Results Folder

- The `results` folder is **static** and cannot be changed during runtime.

### Step 3: Endpoints Path

- Modify paths for both endpoints regarding your use case.

## Running the Application

To run the FastAPI application, follow these steps:

1. **Navigate to the directory** containing `main.py` (your FastAPI entry point).
   
   ```bash
   cd /path/to/your/project
   ```

2. **Start FastAPI in development mode** with the following command:

   ```bash
   fastapi dev main.py
   ```

   This will start the FastAPI application in development mode (it will allow automatic reloading when changes are made).

## FastAPI Documentation

Once the application is running, you can access validation endpoint at:

```plaintext
http://127.0.0.1:8000/validate-<your-use-case>
```

And classification enpoint at:

```plaintext
http://127.0.0.1:8000/classify-<your-use-case>
```

---

### Additional Notes

- Ensure that all necessary files are present and that the application has access to required resources.
- If you run into any issues, please check the logs for detailed error messages.
```

