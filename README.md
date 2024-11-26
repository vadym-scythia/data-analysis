# Data Analysis Project

This README provides the steps to set up a Python virtual environment for this repository, ensuring you can run the scripts and work with the dependencies in isolation.

## Prerequisites
- Ensure you have Python 3 installed on your system. You can download the latest version from [python.org](https://www.python.org/downloads/).
- It is recommended to add Python to your system PATH during installation.

## Setting Up a Python Virtual Environment

1. **Clone the Repository**
   
   First, clone the repository from GitHub:
   ```sh
   git clone https://github.com/vadym-scythia/data-analysis.git
   cd data-analysis
   ```

2. **Create a Virtual Environment**
   
   To create a virtual environment named `venv`, run the following command:
   ```sh
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   
   - On **Windows**:
     ```cmd
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```sh
     source venv/bin/activate
     ```

   After activation, you should see `(venv)` at the beginning of the command line, indicating that the virtual environment is active.

4. **Install Dependencies**
   
   Install the required packages using `pip`. The repository should include a `requirements.txt` file:
   ```sh
   pip install -r requirements.txt
   ```

5. **Run the Scripts**
   
   You can now run the Python scripts in the repository using:
   ```sh
   python <script_name>.py
   ```

## Deactivating the Virtual Environment

Once you are done working on the project, deactivate the virtual environment by running:
```sh
deactivate
```

## Additional Information
- If you add new dependencies, you can update the `requirements.txt` file by running:
  ```sh
  pip freeze > requirements.txt
  ```
- Make sure to activate the virtual environment each time you start working on the project to ensure that all dependencies are available.

