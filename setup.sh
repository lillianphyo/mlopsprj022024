#!/bin/bash

# Navigate to the src directory
cd src

# Step 1: Set up a virtual environment (optional but recommended)
echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Install required Python packages
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Train the LSTM model (if you haven't trained it yet)
echo "Training the LSTM model..."
python model.py

# Step 4: Ensure model directory exists and move the trained model
echo "Creating model directory and moving the trained model..."
mkdir -p model
mv lstm_model.h5 model/lstm_model.h5

# Step 5: Build the Docker image
echo "Building the Docker image..."
docker build -t lstm_flask_app .

# Step 6: Run the Docker container
echo "Running the Docker container..."
docker run -d -p 5000:5000 --name lstm_flask_app_container lstm_flask_app

# Step 7: Run unit tests
echo "Running unit tests..."
python -m unittest discover -s tests

# Step 8: Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate

# Navigate back to the root directory
cd ..

echo "Setup complete. The Flask app is running in a Docker container on port 5000."
