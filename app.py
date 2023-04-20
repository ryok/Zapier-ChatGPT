import sys
import os
import re
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
import gradio as gr


# search = GoogleSearchAPIWrapper()
zapier = ZapierNLAWrapper()


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)


def on_token_change(user_token, state):
    print(user_token)
    # openai.api_key = user_token or os.environ.get("ZAPIER_NLA_API_KEY")
    os.environ["ZAPIER_NLA_API_KEY"] = user_token
    state["user_token"] = user_token
    return state
        

class ConversationBot:
    def __init__(self):
        print("Initializing ChatGPT")
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        self.agent = initialize_agent(
            self.toolkit.get_tools(),
            self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True)

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

        
if __name__ == '__main__':
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Zapier ChatGPT")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.8):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            with gr.Column(scale=0.2, min_width=0):
                clear = gr.Button("Clear️")
        with gr.Row():
            with gr.Column():
                gr.Markdown("Enter your own Zapier API Key. You can get it [here](https://zapier.com/l/natural-language-actions).")
                user_token = gr.Textbox(placeholder="Zapier API Key", type="password", show_label=False)

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        user_token.change(on_token_change, inputs=[user_token, state], outputs=[state])
        demo.launch(server_name="0.0.0.0", server_port=8080)