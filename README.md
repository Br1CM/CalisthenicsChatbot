# CalistenIA
Street workout Agentic RAG with posture correction for different excercises.

As of now, only supporting posture correction as an assistant for:
 - Squats
 - Push Ups
 - Pull Ups
 - Dips

Agentic RAG powered by Llama 3.1 that provides help with sport related topics.

As a prove of concept, the tools were created and implemented to be ready to use both as an simple API and as an app made with Streamlit.

## Running the Project with Docker

**IMPORTANT NOTE**:

The Docker is meant to be only for the webapp created with Streamlit. The API is not meant for the Docker image.

To run this project using Docker, follow these steps:

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone https://github.com/Br1CM/CalisthenicsChatbot.git
   cd <your-repo-directory>
   ```

2. **Set up environment variables**:

   - Create a `.env` file in the root directory.
   - Add your Tavily API key and (optionally) set the host port. Example:
     ```
     TAVILY_API_KEY=your_tavily_api_key_here
     HOST_PORT=3000
     ```

3. **Build and start the containers**:

   - First, make sure you can access the ollama Docker 

   ```bash
   docker-compose up -d ollama
   docker-compose build app
   ```

   - Once the container is started, pull llama3.1 into it

   ```bash
   docker-compose up -d ollama
   docker-compose exec ollama ollama pull llama3.1
   ```
   - restart ollama

   ```bash
   docker-compose restart ollama
   ```

   - Start the app
   
   ```bash
   docker-compose up -d app
   ```

   This will:
   - Start the Ollama service (for Llama 3.1 inference).
   - Start the Streamlit app (or your main application).

4. **Access the application**:

   - Open your browser and go to: [http://localhost:3000](http://localhost:3000) (or the port you set in `HOST_PORT`).

5. **Persisted Data**:

   - Downloaded Ollama models are persisted in a Docker volume (`ollama_models`), so you don't need to re-download them every time.

6. **Stopping the services**:

   - To stop the containers, run:
     ```bash
     docker-compose down
     ```

**Note:**  
- Make sure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- The first run may take a while as it downloads the required models and builds the images.

