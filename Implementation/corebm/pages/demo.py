import os
import streamlit as st

from corebm.pages.task import task_config
from corebm.systems import *
from corebm.utils import init_openai_api, read_json

def demo():
    init_openai_api(read_json('config/api-config.json'))
    st.set_page_config(
        page_title="CoReBM",
        page_icon="📄",
        layout="wide",
    )
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'systems', 'collaboration', 'all_agents.json')
    task_config(task='pr', system_type=CollaborationSystem, config_path=config_path)
