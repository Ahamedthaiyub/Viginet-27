# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./streamlit_app/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py"]
