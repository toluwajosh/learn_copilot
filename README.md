# Learn Pilot

Working codebase for personalized and adaptive learning based on LLMs.

## Usage

- Create Python environment
- Install packages: pip install -r requirements.ini
- Create an environment file `.env` and add `OPENAI_API_KEY=#############################`
- Use the script `add_library.py to add new documents to your library.
- Create `config/params.json` to store information about your library. See example in `config/params_example.json`
- Start the app with `streamlit run main.py`

## TODOs

- [X] Chat with document
- [ ] Show references to document
- [ ] Choose between offline and cloud models
- [ ] Make answers more elaborated with examples
- [ ] Database of students' progress
- [ ] Enable [more complex prompts](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples_chat)
- [ ] Create learning assessments for topics in the document
- [ ] Minimal UI with streamlit
