import json

import tiktoken
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from Implementation.corebm.agents.base import Agent
from Implementation.corebm import AnyOpenAILLM
from Implementation.corebm import format_step, format_supervisions, read_json, get_rm

class Supervisor(Agent):
    """
    The supervisor agent. The supervisor agent prompts the LLM to supervise on the input and the scratchpad as default.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """Initialize the supervisor agent. The supervisor agent prompts the LLM to supervise on the input and the scratchpad as default.
        
        Args:
            `config_path` (`str`): The path to the config file of the supervisor LLM.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        keep_supervise = get_rm(config, 'keep_supervise', True)
        self.llm = self.get_LLM(config=config)
        if isinstance(self.llm, AnyOpenAILLM):
            self.enc = tiktoken.encoding_for_model(self.llm.model_name)
        else:
            self.enc = AutoTokenizer.from_pretrained(self.llm.model_name)
        self.json_mode = self.llm.json_mode
        self.keep_supervise = keep_supervise
        self.supervisions: list[str] = []
        self.supervisions_str: str = ''

    @property
    def supervisor_prompt(self) -> PromptTemplate:
        if self.json_mode:
            return self.prompts['supervisor_prompt_json']
        else:
            return ''
        
    @property
    def supervisor_examples(self) -> str:
        prompt_name = 'supervisor_examples_json' if self.json_mode else 'supervisor_examples'
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            return ''

    def parse(self, response: str, json_mode: bool = False) -> str:
        if json_mode:
            try:
                json_response = json.loads(response)
                if 'new_plan' in json_response:
                    return f"{json_response['correctness']}\n- **Reason**: {json_response['reason']}\n- **New Plan**: {json_response['new_plan']}"
                else:
                    return f"{json_response['correctness']}\n- **Reason**: {json_response['reason']}"
            except:
                return 'Invalid response'
        else:
            return response

    def _build_supervisor_prompt(self, input: str, scratchpad: str) -> str:
        return self.supervisor_prompt.format(
            examples=self.supervisor_examples,
            input=input,
            scratchpad=scratchpad
        )
    
    def _prompt_supervision(self, input: str, scratchpad: str) -> str:
        supervisor_prompt = self._build_supervisor_prompt(input, scratchpad)
        supervisor_response = self.llm(supervisor_prompt)
        supervisor_response_json = format_step(supervisor_response)
        supervisor_response_json = self.parse(supervisor_response_json, self.json_mode)
        if self.keep_supervise:
            self.supervisor_input = supervisor_prompt
            self.supervisor_output = supervisor_response_json
            logger.trace(f'Supervisor input length: {len(self.enc.encode(self.supervisor_input))}')
            logger.trace(f"Supervisor input: {self.supervisor_input}")
            logger.trace(f'Supervisor output length: {len(self.enc.encode(self.supervisor_output))}')
            if self.json_mode:
                self.system.log(f"[Correctness]: {self.supervisor_output}", agent=self, logging=False)
            else:
                self.system.log(f"[Correctness]:\n- {self.supervisor_output}", agent=self, logging=False)
            logger.debug(f"Supervisor output: {self.supervisor_output}")
        return format_step(supervisor_response)

    def forward(self, input: str, scratchpad: str, *args, **kwargs) -> str:
        logger.trace('Running Supervise...')

        self.supervisions.append(self._prompt_supervision(input=input, scratchpad=scratchpad))
        self.supervisions_str = format_supervisions(self.supervisions, header=self.prompts['supervise_header'])

        logger.trace(self.supervisions_str)
        return self.supervisions_str
