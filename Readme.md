# RAG with Human Feedback

This project explores the implementation of the Retrieval-Augmented Generation (RAG) model with human feedback, leveraging OpenAI's powerful language capabilities. RAG is a model architecture that combines retrieval-based and generative approaches to provide rich and contextually relevant responses.

## Getting Started

Follow these steps to get started with the project:

1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-with-human-feedback.git
cd rag-with-human-feedback
```

2. Install the Requirements

```bash
pip install -r requirements.txt
```

3. Insert OpenAI Key

Insert your OpenAI API key in the appropriate configuration file or environment variable. Use the .env_template file and paste the key
```bash
OPENAI_API_KEY="sk-#####################"
```

4. Run the Streamlit Application

```bash
streamlit run main2.py
```

This will start the Streamlit application where you can interact with the RAG model and get the outputs in the testing folder and training folder

5. Label the Data

Ensure your data is appropriately labeled for training.[using training folder] 

### Run the CrossEncoder Notebook for Training

Execute the `crosencoder_training.ipynb` file to train the model.

6. Run the Streamlit Application for Fine-tuning

```bash
streamlit run main3.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
